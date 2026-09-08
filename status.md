# MCOG status

**Living handoff doc.** Read this first, then follow the pointers:

| Where | What |
|---|---|
| `docs/STRUCTURE.md` | repo map, layering rules, gates, artifacts, environment, tooling |
| `docs/TODO.md` | open tasks, next action for each |
| `docs/PROGRESS.md` | dated session log: question -> what ran -> result -> next |
| `docs/stage1_env_spec.md` | Stage 1 derivations and memo citations |
| `docs/hierarchical_contact_rich_manipulation_research_plan.pdf` | the memo everything cites |

This file keeps current state, the plan, the Stage 0 result, and the two reference sections
worth having in front of you at all times: the gotchas and the physics numbers.

---

## Current state (2026-09-08 late — SWEEPS C+D IN FLIGHT)

### START HERE — what a fresh session must know, in one screen

**Sweep C (job `45467495`, 12 cells) and Sweep D (job `45467480`, 6 cells) are
RUNNING**, both at commit `87b7d3f` with `GIT_DIRTY=no`. Gates
**42/27/290/172/18**. Sweep D lands first (~7-13h); Sweep C ~16h/cell.
Full detail in `docs/PROGRESS.md` (2026-09-08 late).

1. **A pre-launch audit found one live bug and it is fixed.** D2 moved the
   still-GUARD to a latched 2cm displacement bound but left `_her_arrived`'s
   sticky `object_disturbed` flag on the instantaneous velocity test, so
   `step()` and the HER relabel scored the same transition differently. On
   `g_two_disp` the flag latched at **median tick 2 of 400 in 45% of episodes**,
   switching HER off for **41.9%** of all ticks; now **0.2%**. `g_one` was never
   affected (its object never moves), which is both why its **0.604** stands and
   why the smoke could not have caught this.
2. **The gate that should have caught it was vacuous.** "Rollout reward ==
   relabeled reward" drove RANDOM actions, where no transition is ever a success
   and both paths agree on 0.0. The replacement constructs the discriminating
   state and fails without the fix. **Assume any regression test that never
   reaches the success branch is not testing the success branch.**
3. **Eq 35 was under-reported ~3x.** The diagnostic eval consumes **2.09x**
   (Sweep C) and **1.78x** (Sweep D) the TRAINING steps. Real totals **~89M** and
   **~17M** environment interactions. Report those, not the training budget.
4. **The curriculum advance is measured, not hoped for.** Level-0 goals demanding
   more than the 22.5deg tolerance: **0.492**, so local success ceilings at ~0.51
   against the corrected 0.4 threshold (0.6 was unreachable). Smoke 45449813 hit
   **0.406 at 180k** — the first advance costs ~7.5% of a 2.4M budget.
5. **`ctl`/`raw`/`obs_v1` are unaffected by the fix** — it is inside
   `if self.template == "recontact"`. Sweep C's four floors regenerated
   bit-identical.

### WHAT TO WATCH, and what to do about it

**Sweep D — score BY HAND, per arm, against `PINS.<arm>.txt`.** The two arms
train on different tasks, so `finalize.sh` (which assumes one common protocol) is
deliberately not wired up. The preregistered instruction stands: **if
`g_two_disp` returns 0.000, the next move is task design, not budget.** Do not
re-run it larger on a flat zero curve.

**Sweep C — the curriculum DOES advance; both smokes finished and settled it.**
Threshold 0.4 advanced level 0 -> 1 at **179,805 steps** (7.5% of a 2.4M
budget); threshold 0.6 stayed at level 0 with a local max of **0.406**, which is
the measured ~0.51 ceiling refusing to reach 0.6. Both runs hit identical local
(0.406) and full-task (0.281) maxima, as bit-identical runs should. After
advancing, rollout success dropped 0.340 -> 0.260 because level 1 is harder --
the curriculum doing its job.

**What is still unproven is levels 1 -> 2 -> 3**, which carry more crossings.
Level 1 local success sits at 0.22-0.28 and has to climb back to 0.4. At level
0's pace (~180k per rung) four levels cost ~550-750k of 2.4M, but the later
rungs are harder. **So `eval/curriculum_level` at the 600k rung is still the
number to watch** -- it should read 2 or 3, not 0 or 1. Mitigation, designed and
NOT built:
a step-based fallback in `domains/contact/callbacks.py` — advance on
`local >= threshold` OR once a level consumes its share of the budget, defaulting
to `None` for bit-identical behaviour. Four rungs mean this surfaces at 25% of
the cost rather than at 2.4M.

### THE PATTERN BEHIND ~23 DEFECTS IN A MONTH — apply it before adding anything

**The same quantity is computed in two places and nothing forces them to agree.**
`obs()` vs `her_buffer`'s divisor; rollout vs relabeled reward (v18, again at
v32, again as the Gamma arrival bug, again as `w_prog`, **again 2026-09-08 as the
guard-vs-HER disturbance criterion**); `step()` vs HER's arrival; launcher vs
scorer vs floor protocol; commanded face vs live normal; `achieved_goal` vs
`desired_goal`'s normalizer. Every fix that stuck has the same shape: **collapse
the copies, then gate the regrowth.**

Collapsed so far: `domains/contact/keys.py` (was five copies of the interface-key
list), `_ray_passes_portal` (two portal checks), `pins_board_v2.sh` /
`pins_gamma_ladder.sh` (retyped protocols), and `_max_disp_cm` as the single
reading of "the object moved".

**The remaining gap, and it has now cost twice:** gates are strong at "does the
code do what it says" and weak at "did this flag do anything at all". The
longest-lived bugs were all silently inert or vacuously green. **A gate that
cannot fail is worse than no gate**, because it is counted.

### AUDIT RESULTS THAT PASSED — the baseline a re-check should reproduce

- All six scoring digests reproduce their floors: `ctl`/`raw`/`obs_v1`
  `98e24e99890f`, `count` `b99150951ac7`, `g_one` `60c119c9ef02`, `g_two_disp`
  `3b54b18fe084`.
- Band curriculum level 3 == the benchmark exactly (med 17.5cm, p90 36.3, max
  49.3, crossing 0.483); ramp 0.00 -> 0.13 -> 0.34 -> 0.48. **`curriculum_leaks`
  = 0** everywhere. **0 of 72 benchmark episodes blocked.**
- `xi_gamma_mode=count` changes exactly the face one-hot (slots 3-6) to a
  constant; the active-finger slot is untouched.
- Sweep D has **0/300 episodes satisfied at t=0** on both arms; 54% of
  `g_two_disp` is the grasp-to-grasp case.

### TWO LIMITATIONS, recorded not hidden

- **`count` bundles three changes** (xi encoding, `guard_contact_count=1`,
  `mask_inactive_finger=false`), inseparable in principle — a count interface
  with the second finger masked is not one. So a `count` WIN is attributable, a
  `count` LOSS is not.
- **The count guard is inert on the masked arms** (max contacts 1 on 200/200, 0
  `forbidden_contact`), so `score_rungs.sh`'s two-way pass returns identical
  numbers for `ctl`/`raw`/`obs_v1`. That is what makes `(count - ctl)` a clean
  paired comparison, and also means 9 of 12 cells are re-scored for a known
  answer.

---

## Current state, previous entry (2026-09-08 — SWEEP B READ, PHASE 1 BUILT)

### START HERE — what a fresh session must know, in one screen

**Sweep B is scored and read. Phase 1 is built, gated and committed. Sweep C and
Sweep D are written and NOT submitted.** Commits `007e5b1` -> `c2ac8fe`.
Gates **42/27/279/172/18**. Full detail in `docs/PROGRESS.md` (2026-09-08).

1. **Push converges for `ctl` and for nothing harder.** `ctl` is flat over the
   last 600k (-0.007); `widecone` gained +0.069 there. All four arms fire the
   preregistered "not converged at 2.4M" branch against the 1.2M rung. **The arm
   ordering FLIPPED between 1.8M and 2.4M** — `widecone` went 2nd -> 3rd -> tied
   -> 1st. A sweep read at one budget can rank arms backwards; four rungs are
   mandatory and that is now in both launchers.
2. **PUSH IS NOW TOO FLAT TO CALIBRATE ON.** `ctl`'s distance spread went
   0.278 -> 0.194 -> **0.083** across the rungs, and at 2.4M success *rises*
   with distance. Distance is dead as a `p_hat` feature on this board. What
   survives is ORIENTATION (0.914 inside tol vs 0.795 must-rotate) and a harder
   protocol (0.694 on cone 90). **This makes the Stage 1 ladder the critical
   path, not just the long pole**, and it is why `contact_descriptors` is built
   on orientation.
3. **obs v2's penalty is real AND the arm was two keys.** No cell has ever
   trained `obs_version=2` alone. The defect is in `normalize_goal_keys`:
   `VecNormalize` keeps a SEPARATE `RunningMeanStd` per key while
   `achieved_goal` and `desired_goal` are the same quantity measured twice —
   std ratio 1.70-1.80 on `cos(theta)`, so a perfectly achieved heading shows the
   network a residual up to 1.24 that CHANGES SIGN inside the goal window. It
   also whitens a unit-vector pair, which `train_contact`'s own comment forbids.
   **This also explains the template disagreement**: recontact's goal key is 2-D
   with no (cos, sin) pair, so obs v2 measured free there. INFERRED, not proven —
   Sweep C's `obs_v1` arm is the decisive test.
4. **The replication check PASSED EXACTLY.** Sweep A and Sweep B checkpoints are
   bit-identical (max |dW| = 0 on 32 tensors, 3 seeds) at both 600k and 1.2M. The
   apparent 0.826 -> 0.806 miss is a snapshot artifact: Sweep A's headline was
   `model.zip`, a few gradient steps past the callback. **So the benchmark's
   resolution is 0.021 — one episode in 48 — and a "final" checkpoint is not the
   same number as a rung checkpoint at the same nominal step.**
5. **`slurm/_run_cell.sh` still had the broken `-o "%A_%a"` last-task check** —
   the bug behind 621 orphaned staging files. `submit_sweep.sh` fixed its own
   INLINE copy on 2026-09-04; the shared one was never touched. Fixed.

### THE PATTERN BEHIND ~22 DEFECTS IN A MONTH — apply it before adding anything

**The same quantity is computed in two places and nothing forces them to agree.**
`obs()` vs `her_buffer`'s divisor; rollout vs relabeled reward (v18,
reintroduced at v32, again as the Gamma arrival bug, again as `w_prog`);
`step()` vs HER's arrival; launcher vs scorer vs floor protocol; commanded face
vs live normal; and `achieved_goal` vs `desired_goal`'s normalizer. Every fix
that stuck has the same shape: **collapse the copies, then gate the regrowth.**

Applied proactively 2026-09-08: `domains/contact/keys.py` replaces five copies of
the interface-key list, `_ray_passes_portal` replaces two portal checks,
`slurm/pins_board_v2.sh` and `slurm/pins_gamma_ladder.sh` replace retyped
protocols. The remaining gap is that gates are strong at "does the code do what
it says" and weak at "did this flag do anything at all" — the longest-lived bugs
(`Monitor` never applied, the finalize trigger never firing, `w_prog` absent from
80% of every batch) were all silently inert.

### BOARD V2 — pinned, floored, and five defects caught before any GPU

`slurm/pins_board_v2.sh` (push) and `slurm/pins_gamma_ladder.sh` (recontact) are
the ONLY copies. Two files because `ContactEnv` REFUSES push-only keys on
recontact rather than ignoring them, so sourcing the wrong one is a hard error.

90x60, 3 rooms, 13cm doors offset 34cm in y, `portal_goal=false`,
`require_settled=true`, cone 75, `same_room_goal_prob=0.5`, and **full START-pose
randomization (`object_theta_spread_deg=180`)**. The GOAL heading window stays at
45deg: push produces a median 1.8deg of rotation per episode, so a uniform goal
heading would be unreachable in ~84% of episodes. Arbitrary reorientation is
`pivot`'s job, and pivot is not implemented.

**TWO PREDICATES, ONE EVAL, ONE DIGEST, NEVER POOLED.** The pose goal is what the
policy trains on (well-posed under HER); "entered the destination region" is what
a crossing edge in a route has to do. Training on the portal predicate would put
the rollout reward and ~80% of every relabeled batch on different objectives, so
it is applied at EVAL only. `tools/split_floor.py` prints both across three
strata.

What the floors caught, all before any GPU:

- **The eval bin edges were wrong.** v1's `[3,6,9,12]` collapsed every crossing
  into "12+" — 11 of 48 in-scope on a 50/50 board. `[3,8,15,25,35]` gives 60
  in-scope at 62% crossing. Pinned beside the numbers.
- **`entered_dst` was trivially true on same-room episodes** — a floor of 1.000
  for doing nothing. Now `None` where the question is not asked.
- **Full pose randomization made the sampler LEAK** (3 of 400 at the old
  256-attempt cap; a leak silently trains a different task). Cap 1024, now gated.
- **The Gamma ladder had to become DIRECTIONAL.** Floors were g_one 0.583 /
  g_two_disp 0.396 with init from all four classes, and g_one was *still* 0.542
  after excluding same-count pairs. **MEASURED: releasing a contact is not a
  skill (floor 0.542); acquiring one is.** Both rungs are acquisitions now.
- **The count abstraction's cost, recorded not hidden:** pinch <-> pivot is a real
  transition that contact count cannot express, so it reads as a no-op.

**THE OFFSET DOOR EARNED ITS KEEP.** The crossing floor was the expected problem
(0.271 overall / 0.42 in the crossing bins on board v1). Offsetting the doors
34cm in y drops it to **0.000-0.162**, so the crossing metric is readable.

| arm | digest | pose >=3cm | entered-dst |
|---|---|---|---|
| `ctl` / `raw` / `obs_v1` | `98e24e99890f` | 0.000 | 0.000 / 0.000 / 0.162 |
| `count` | `b99150951ac7` | 0.000 | 0.000 |
| `g_one` | `60c119c9ef02` | 0.208 | — |
| `g_two_disp` | `3b54b18fe084` | 0.000 | — |

### The two sweeps, at a glance — SUBMITTED 2026-09-08 late

| sweep | cells | budget | arms | decides |
|---|---|---|---|---|
| C `slurm/submit_sweep_c.sh` | 12 | 2.4M, 4 rungs | `ctl`/`count`/`raw`/`obs_v1` | the action space, the Gamma encoding, the observation version |
| D `slurm/submit_sweep_d.sh` | 6 | 1M, 4 rungs | `g_one`/`g_two_disp` | whether the contact interface is acquirable at all |

**All four Sweep C arms run obs v2 with `obs_v1` as the CONTRAST**, because obs
v2 is the version the project intends to keep — measuring the interface arms at
v1 would mean re-running them the moment it is adopted. `normalize_goal_keys` is
OFF on every arm, for the measured reason in item 3.

**Sweep D's zero has an instruction attached, on the record before the numbers:**
if `g_two_disp` also returns 0.000, the guard's FORM was not the wall either and
the next move is task design, not budget. Do not re-run it at a larger budget on
a flat zero curve.

Eq 35, MEASURED and corrected 2026-09-08 late: **~89M (C) + ~17M (D)**
environment interactions, because the diagnostic eval costs 2.09x/1.78x the
training steps. Measured 16.0h/cell at 2.4M against a 30h wall.

---

## Current state, previous entry (2026-09-04 late — PHASE 0 DONE, SWEEP B IN FLIGHT)

### START HERE — what changed in the last session, in one screen

Five free measurements (no GPU, ~0.5M env steps) reshaped the plan and cut it in half:
**30 cells / 55.2M -> 15 cells / 27.6M.** Full numbers in `docs/PROGRESS.md` (entry dated
2026-09-04 later); the decisions they license are `docs/TODO.md`'s **DECISIONS TAKEN** and
**PREREGISTERED VERDICTS** sections, both written before the sweeps and not to be
re-litigated.

The five things a fresh session must know:

1. **Gamma is now `{free, one-contact, two-contact}`.** The commanded contact FACE is
   retired (D1). Eq 13's positional interface is SATISFIABLE — fingers placed exactly on
   target score arrived **80/80** for all three classes with the object moving 0.000cm — so
   the 12-cells-at-0.000 result was never bad geometry. The 4-way face one-hot is retired
   because it breaks at the memo's own T-shape and `strict` face-guarding scores **0.090**.
2. **The still-guard was the blocker, and it had switched HER OFF.** `object_disturbed`
   fires on **237 of 240** scripted episodes at a median tick 8 of 200, and it is STICKY and
   gates `_her_arrived` — so ~96% of an episode's transitions produce relabels that can never
   pay. That is the only account offered for v34's "39.4% of Gamma episodes never got closer
   than their start". `guard_terminates=false` would NOT have fixed it.
3. **The guard becomes a DISPLACEMENT bound, eps = 2cm, latched running max** (D2). Measured:
   at a 2cm/s action scale the peak disturbance is a median 0.62-0.76cm (p90 1.03-1.19), so
   eps=2cm leaves **1.000** of ticks eligible; at 20cm/s even eps=3cm leaves 0.33-0.43, so
   **the displacement form and the low action scale are complementary, not substitutes.**
4. **`push_cone_deg` 30 -> 75**, from a measured p90 of 74.2deg (D3). +/-180 is not
   supportable (1.4%). And **most "behind the contact" reachability was bought by walking to
   another face**: the 120-180deg bin goes 0.176 -> 0.014 under `guard_face=adjacent`, where
   `wrong_face` becomes the largest exit at 42%.
5. **A LATENT SAMPLER BUG, found by the board probe.** `_sample_push_edge_reverse` — the
   sampler `curriculum_mode=band` selects, i.e. the one every sweep since v32 uses, Sweep B
   included — never checks that the object->goal ray passes the doorway.
   `_sample_goal_in_push_cone` does (gym_env.py:727-733); the reverse sampler does not.
   `portal_goal=true` hid it by making the goal BE the doorway. **No shipped number is
   affected** (every sweep pinned `portal_goal=true`), but board v2 cannot launch without the
   fix — see `docs/TODO.md` ORDER OF WORK 2a.

### The next two sweeps, at a glance

| sweep | cells | budget | arms | decides |
|---|---|---|---|---|
| interface + abstraction | 9 | 2.4M, 4 rungs | `ctl` / `count` / `raw` | the action space and the Gamma encoding everything downstream uses |
| Gamma ladder | 6 | 1M | `g_one` / `g_two_disp` | whether Gamma is acquirable at all, and whether the guard's form was the blocker |

Both on **board v2** (D4: 90x60, 3 rooms, 13cm doors offset 34cm in y, `portal_goal=false`,
`require_settled=true`, cone 75). Gamma reuses v34's `gamma_free`/`gamma_init` 0.000 columns
as references, saving 6 cells. **PPO and its `sac_dense` control are deferred** — 6 cells
answering a replication that gates no design decision. The PPO floor branch is done and
launch-ready.

### THE RULE THAT PAID THIS SESSION — apply it before adding any arm

**Before adding a sweep arm, ask whether a scripted controller or a frozen checkpoint can
answer it.** Half the arms proposed last session died to that question, each for minutes of
CPU: the Gamma tolerance rung (satisfiability is 100%), the Gamma action-scale arm (priced
directly, 0.050 vs 0.425), the strict face-guard arm (0.090), `g_soft` and `g_pos_slow`,
the two-contact positional arms, and `push_cone_deg` itself (measured, not swept). Plus a
sampler bug that would have wasted an entire round, and a horizon bug that would have
returned 0.000 for a reason unrelated to its hypothesis (`v_max=2` x `horizon=200` gives only
16cm of travel against a 8.0-16.1cm spawn radius — the arm needs `horizon=400`).

New tools, all committed: `tools/probe_gamma_feasible.py` (parts A-F: satisfiability,
scripted reachability, displacement, gentle-approach, contact-count, guard-form vs HER
eligibility), `tools/probe_reachable.py`, `tools/probe_board_v2.py`,
`tools/score_v34_faceguard.sh`. Commits `d64bf71` -> `a82d90f` -> `b0fe6be` -> `3618a80`.
Gates **40**/27/243/172/18.

### THE FREEZE IS NARROWER THAN EARLIER ENTRIES SAID

`finalize.sh` runs only `tools/score_sweep.py` (which runs `eval_contact.py`) and
`tools/render_best.py`. **Those two plus what they import are the frozen set** until Sweep B
scores. Everything else — new tools, launchers, floor scripts, `test_code.py`,
`tools/make_untrained_ckpt.py`, and the entire Stage 1 ladder build — is fair game and was
worked on during Sweep B without touching it.

### The face guard is priced, and `adjacent` is affordable

`tools/score_v34_faceguard.sh` (job 44432898, 4 minutes), Sweep A's 9 cells, own PINS with
`guard_face` flipped and nothing else. Zero-shot, so read as a DELTA, never pooled.

| arm | guard off | `adjacent` | `strict` |
|---|---|---|---|
| a1 | 0.736 | **0.653** (-0.083) | 0.090 (-0.646) |
| a2 | 0.701 | 0.646 (-0.055) | 0.090 (-0.611) |
| a3 | 0.681 | 0.639 (-0.042) | 0.083 (-0.598) |

Digests `780f93cc40d9` / `ad7e75719a0f`. `wrong_face` 11-17% at adjacent, **81-82%** at
strict. **The arm ordering does NOT invert this time** (a1 ahead under both), so v32's
inversion was a curriculum artifact and that worry retires. Eq 40 CAN be enforced at
`adjacent`; it stays off in the interface sweep only so the arms are not floored (D5).

---

## Current state, previous entry (2026-09-04 — v34 SCORED, SWEEP B IN FLIGHT)

### SWEEP B IS RUNNING — job 44379812, 12 cells, submitted 2026-09-04

`ctl` / `obsv2` / `widecone` / `spread` x seed{0,1,2} at **2.4M**,
`ckpt_freq=600000`, 30h wall, ~17.5h/cell expected. Rungs land at 600k (v33's
budget), 1.2M (Sweep A's), 1.8M and 2.4M, so **every price is a curve, not a
point** — that is the whole design, and the reason is in Result 1 below.

**First sweep in eight with clean provenance:** all 12 cells record
`GIT_COMMIT=4a1bfd9`, `GIT_DIRTY=no`, and a zero-byte `uncommitted.diff`.

Full arm rationale, the omissions and their measured reasons, and the
preregistered verdicts are in `slurm/submit_sweep.sh`'s header — written before
the numbers land, and not to be re-litigated after.

**HOW IT SCORES ITSELF.** The last task standing submits both
`slurm/finalize.sh <dir> sweepB` (model + model_best on the common protocol) and
`tools/score_v35_rungs.sh <dir>` (the three intermediate rungs, plus `widecone`
and `spread` each scored on their OWN distribution, whole-sweep, because the
DELTA against `ctl` on the loosened task is the quantity of interest). The
trigger now works but has fired exactly once, so **check for
`logs/sweep_44379812/slurm_logs/` and run both by hand if it is absent.**

Common benchmark: the FACE-CENTRE protocol, digest **`249434216cd2`** — the one
every archived v32/v33/v34 push figure sits on, so `ctl` anchors with no
transfer step. Floors, regenerated by `tools/make_v35_floor.sh` (deleted 2026-09-08; see git history) before the sweep
and bit-identical on a full re-run:

| cell | digest | floor `>=3cm` |
|---|---|---|
| `ctl` | `249434216cd2` | **0.042** (reproduces `logs/eval/v34_floor/a1_v1_centre`) |
| `obsv2` | `249434216cd2` | 0.021 (same digest, different floor — the untrained net reads a different vector) |
| `widecone` | `b21b11ecf4fc` | 0.000 |
| `spread` | `5a24875f15c4` | 0.000 |

### V34 RESULT — all 24 cells, scored 2026-09-04

**`finalize.sh`'s auto-trigger fired for the first time ever and killed its own
scorer**: it writes `slurm_logs/` into the sweep dir it then scores, and
`score_sweep.cell_dirs` globbed `*/`. All three jobs died on
`AttributeError: 'NoneType' object has no attribute 'group'` with the eval dirs
created and empty. Fixed and gated (`static` 36 -> 38).

#### Result 1 — THE BUDGET IS THE EFFECT, and it is 3-6x either treatment

Goals >=3cm. Paired per-episode bootstrap CIs over the 48 in-scope episodes —
valid because the env digest fixes the same 60 initial states for every arm.

| contrast | ALONG `646ba4ae1fd4` final / best | CENTRE `249434216cd2` final / best |
|---|---|---|
| spawn `a2 - a1` | -0.035 / +0.021 | -0.042 / **-0.090** |
| obs v2 `a3 - a2` | -0.021 / **-0.132** | -0.062 / +0.021 |
| both `a3 - a1` | -0.056 / **-0.111** | **-0.104** / -0.069 |
| **budget 1.2M - 600k, within arm** | — | **a1 +0.208, a2 +0.132, a3 +0.125** |

Bold = 95% CI excludes zero. **The budget is the only effect significant on
every arm and in the same direction.**

**Levels.** A1 on its own protocol: **0.826** final (0.792/0.833/0.854) / 0.764
best, against 0.618 at the 600k rung and v33 `ctl`'s 0.674. Common protocol,
final: A1 0.736 > A2 0.701 > A3 0.681.

1. **The consequential branch fired.** A1 @1.2M is materially above 0.674, so
   **every v33 scaffold price was read at an unconverged budget.**
2. **Continuity held, so the v33 table survives as a 600k table.** A1 @600k =
   0.618 (0.604/0.667/0.583) vs v33 `ctl` 0.674 (0.604/0.750/0.667) —
   indistinguishable at n=3, same digest. Reading them as *converged* prices is
   what does not survive.
3. **THE ARM ORDERING MOVES WITH BUDGET.** Centre protocol: A2 leads at 600k
   (0.653 vs 0.618), A1 leads at 1.2M (0.826 vs 0.785). **A sweep read at one
   budget cannot rank arms in this task.**
4. **Neither treatment is adopted.** A1's config (obs v1, face-centre spawn)
   stays the push default.
5. **The 2026-09-03 diag reading — "the along-face spawn looks like it HELPS" —
   was WRONG**, in exactly the way CLAUDE.md's opening rule predicts: A1 and A2
   sat on different digests, so those columns were never subtractable. **Do not
   quote a diag eval across digests again.**
6. **The spawn trades torque authority for contact retention.** A1 loses contact
   on 7% of episodes against A2/A3's 14-15% (an along-face contact starts nearer
   the corner; v18 measured 62% of first breaks as tangential slides). But A1's
   failures are FAR (`min_dist` median 4.79cm, 21% within 1cm) and A2's are
   CLOSE (1.06cm, 49%). Different failure families, not more or less failure.

#### Result 2 — Eq 13's Gamma goal is 0.000 on 12 of 12 cells

Digest `5dff6e0afd4a`, floor 0.000, 48 episodes/cell (worst-fingertip distance
has a 15.8cm median at reset, so the 0-3cm bin is unfillable). All four arms,
three seeds, **both** checkpoints: 0.000. Pooling hides nothing — zero overall
is zero in every class.

| arm | terminations | `min_dist` median | Q gap |
|---|---|---|---|
| `gamma_free` | horizon 65% / disturbed 35% | 7.97cm | +0.42 |
| `gamma_init` | horizon 56% / disturbed 44% | 6.94cm | +0.94 |
| `gamma_init_noclip` | horizon 53% / disturbed 47% | 7.84cm | **+727.76** |
| `gamma_init_shaped` | disturbed 62% / horizon 38% | 7.95cm | **+210.18** |

- **Zero of 576 failures came within 1cm** — family "never got close", and
  `final_dist - min_dist` is 5.2-11.6cm, so it wanders away after closest
  approach.
- **`model_best` is the untrained snapshot on every gamma cell**: identical
  `min_dist` median (8.40cm) and identical termination counts across four arms
  that differ in reward and clipping. Nothing ever improved.
- **`target_clip=null` blows the critic to +727** above realized return against
  `gamma_init`'s +0.94, so the `noclip` control earned its three cells — but the
  clamp is not what blocked Eq 13, because the clipped arm is also 0.000.
  Shaping tames the critic to +210 and rescues nothing.

**Verdict, as preregistered: the 4-way conjunction is not acquirable as posed,
and the next move is task design, not budget.** Staged or sequential fingertip
goals, looser per-finger tolerances, or implementing `pivot`. **Do not re-run
gamma at a larger budget on a flat zero curve.**

#### Result 3 — obs v2 IS free for recontact, and the digest that said otherwise was lying

`base_v2` = **0.967** all bins / 0.958 on `>=3cm` (0.979/0.958/0.938), 97%
arrived, no floored bin, floor 0.033, digest `1ecc01e69a3d`. The archived
`recon_base` is 0.978 at `a78252c0a0a6`, so the preregistered comparison was not
licensed. **Checked instead of assumed** — rebuilt an untrained cell at
`rich_obs=false` with everything else identical:

```
rich_obs=false, obs v1   digest a78252c0a0a6   floor 0.0333
rich_obs=true,  obs v2   digest 1ecc01e69a3d   floor 0.0333
60 per-episode d0 values IDENTICAL
```

So the digest moved on `rich_obs` alone and the two protocols are the same task:
**obs v2 costs recontact 0.011, inside the 0.017 seed sd.**

**OPEN CLASSIFICATION ERROR, recorded and deliberately not fixed.** `rich_obs`
changes what the policy READS, not what the task IS, so by the repo's own
INTERFACE/TASK rule it is the **twelfth interface key**. Moving it collapses
`a78252c0a0a6` and `1ecc01e69a3d` into one digest — correct, but it relabels
every stored score, so it is a decision, not a drive-by fix.

**obs v2's verdict now DISAGREES between templates** — free for recontact,
-0.132 on `model_best` for push. That is why it is an arm in Sweep B rather than
an adopted default.

### Phase 0 (P1-P9) — all landed 2026-09-02/03, full detail in `docs/PROGRESS.md`

Gates went 30/27/141 -> **36/27/243**/172/18 across the phase.

- **obs v2** — `ObsScales` is the one place any observation divisor lives.
  Fixed three measured scale bugs (angular velocity had **no divisor**, range
  +/-3.27; force divided by a **1000.0** fallback against a p99 of 284; goal by
  the board extent 50 where a goal never exceeds 22.2cm) and two live recontact
  bugs that **masked each other** — the goal tail differenced an object-frame
  target against a world position (mean -0.492 over a range of 0.21), and the HER
  goal-tail patch was push-only so recontact's tail was stale on ~80% of every
  batch. `obs_version` is an INTERFACE key, so `249434216cd2` never moved and
  v1-vs-v2 is one benchmark. **Relative heading is the strongest single predictor
  of success at 0.565** — third independent confirmation that orientation, not
  distance, is push's real axis.
- **P1 `normalize_goal_keys`** — VecNormalize over the two goal keys only,
  closing a measured **34x** conditioning gap (goal positions std 7.79/7.55/4.64/
  3.93 against a median of 0.227). `observation` excluded on purpose: 22 dims are
  unit-vector pairs or one-hots that whitening destroys. Safe with HER because
  `compute_reward` runs on RAW arrays before normalization. Stats travel with the
  checkpoint and `eval_contact` refuses a mismatch **in both directions**.
- **P2/P3 the Gamma arrival bug** — `step` scored arrival against finger **L's**
  slot while measuring the **active** finger, so a state perfectly achieving the
  6-D goal was called arrived on **254/500** resets. Now one shared definition,
  the tolerance rides per transition in `info["gamma_tol"]`, and `gamma_goal` has
  gate checks where it had **zero**. Verified 500/500. **The ~63 GPU-hours of
  pre-fix Gamma arms need re-running, not re-scoring — that is job 44180185.**
- **P4 the dense reward vector** — `w_guard` per-outcome, `w_hold`
  (contact-AND-face), `w_settle` (proximity-gated), `w_prog` (potential-based,
  passes a round-trip-sums-to-zero test that absolute `w_d` fails at -9.0),
  `w_arrive_pos`. All inert by default, **all credit-metered**: at `w_hold=0.02`
  a 2k-step run hit **ep_rew_mean 4.57 with success 0.000** at full horizon —
  "hold and stall" worth 46% of the arrival bonus.
- **P5 the PPO path** — `rl_algo=sac|ppo`, **named `rl_algo` because
  `config/algo/` is nav's Hydra config group**. `net_arch=null` resolves to
  `[256,256]` for BOTH algos: SB3's PPO default is `[64,64]`, a ~16x capacity gap
  that memo sec 9 forbids and that would have confounded the comparison before it
  ran. `eval_contact` labels the PPO column `[V, not Q]`.
- **P6-P9** — the recontact launcher rewritten with the full v33 apparatus;
  **every recontact mp4 was 0 bytes** (509px odd axis, libx264; push's board
  gives 420, which is why only recontact broke) plus the goal marker drawn in the
  wrong frame; `push_spawn_along_frac` (the face centre produces **exactly zero
  torque**); `angular_drag_arm_cm` default 3.12 (6.0 exceeds the attainable
  5.83cm); the inert nested curriculum deleted; and the floor regenerated at
  **0.042**, so Bar 1 stands.
- **Three digest traps, same family.** Adding `RewardWeights` fields moved every
  digest (`249434216cd2 -> 436dee0952c5`) because the digest is a sha1 over
  `repr()`; a new env kwarg did it again (`-> e35ceab30ae5`). Fixed by a frozen
  legacy-field `__repr__` and `stamp_omit_if_default`. **Only ever safe for a key
  whose default is bit-identical — check by replaying, do not assume.**

### The pre-launch audit, 2026-09-03 — Bar 2 came free, and 9 problems were fixed

**BAR 2 IS MET, ZERO-SHOT.** v33's frozen `ctl` checkpoints re-scored under
`require_settled=true`: **0.625** mean on goals >=3cm (0.583/0.708/0.583) against
**0.674** position-only and a **0.000** settled floor. The price of settling is
**0.049**; the "expensive half" expectation was wrong. **Phase B is unblocked
with no training arm.** `logs/eval/v34_bar2/` + `tools/bar2_zeroshot.sh`. Valid
on a frozen checkpoint because `require_settled` changes the arrival TEST, not
the observation, the action space or the reset sampler — the same 60 initial
states under both digests. The position column reproduces v33's 0.674 exactly,
which doubles as the regression check on the whole session.

**ORIENTATION IS MOSTLY FREE, AND THAT DEFLATES THE HEADLINE.** New
`eval_contact.orientation_report`. On `ctl_s1`: **42 of 60 goals (70%) are
already inside the 22.5deg tolerance at reset**, success 0.762 there against
**0.500** on the 18 that must rotate, and mean |dtheta| *increases* by 1.36deg
over an episode (+14.02 untrained). Build `contact_descriptors` on orientation,
and read Sweep B's rotation arm against the 70%/0.500 split, never the pooled
number.

Nine problems found, all fixed and gated (detail in `docs/PROGRESS.md`):
`w_arrive_pos` paid **per tick** on the HER path (measured 300 of implied Q
against `goal_reward=10`, on 80% of every batch) -> refused with `use_her`, and
Sweep D's shaped arm uses `w_prog=0.1` instead; the blanket target_clip refusal
**stopped `base_v2` from starting at all** by rejecting recontact's own archived
baseline -> split by shaping SIGN, since negative-only shaping keeps
`Q* <= goal_reward` and only the lower clamp is violated; `w_prog` was silently
absent from 80% of batches; `make_untrained_ckpt.py` hardcoded `push`, so the
Gamma floor would have been a push floor; a misspelled `w_guard` key was silently
free; `gamma_init_shaped` confounded shaping with `target_clip` -> added
`gamma_init_noclip`, 9 -> 12 cells; Sweep D had no floor at all; and `set -e`
with a bare `rc=$?` meant a failing last task skipped `finalize.sh`.

**Known gap:** no PPO floor exists — `make_untrained_ckpt.py` builds a SAC net
and `eval_contact` loads by `rl_algo`, so Sweep C needs a PPO branch there before
it can be scored. Does not affect A or D.


## Superseded state (v33 scoring; still current for push)

**Stage 0 is scientifically closed and the repo matches it.** Preregistered question
answered, mechanism confirmed, audit fixes landed, vocabulary renamed, numbers re-verified
bit-identical at every step.

### THE PUSH SUCCESS BAR — decided 2026-09-01, in `docs/TODO.md`, do not re-litigate

Push has been re-swept six times partly because "good enough" was never written down.
**Bar 1 (LADDER READY):** on the v32 benchmark, mean success on goals >=3cm **>= 0.40**, on
>=2 of 3 seeds, under both `model` and `model_best`, **and no distance bin at 0.00**. The
per-bin floor is the real requirement — Stage 0's ladder failed exactly where strata were
floored at zero. 0.40 is ~10x the measured untrained floor of 0.042.
**Bar 2 (COMPOSITION READY):** Bar 1 plus `require_settled=true`, because a successor option
cannot start from a moving object. **MET 2026-09-03, ZERO-SHOT — no training arm needed.**
v33's frozen `ctl` checkpoints re-scored under settled arrival: **0.625** mean on goals
>=3cm (0.583/0.708/0.583) against **0.674** position-only and a **0.000** settled floor.
The price of settling is **0.049**; the "expensive half" expectation was wrong.
`logs/eval/v34_bar2/` + `PROTOCOL.md`, digest `fdc2a41dc665` (settled) vs `249434216cd2`
(position). **PHASE B IS UNBLOCKED.** Valid on a frozen checkpoint because
`require_settled` changes the arrival TEST, not the observation, the action space or the
reset sampler — so both digests draw the same 60 initial states.
**Neither bar gates the ladder BUILD**, which needs push graded, not good — and it already is.
The axis it is graded on is **orientation, not distance**: success is flat at 0.833 across
the 3-6/6-9/9-12cm bins on ctl_s1, while 70% of benchmark goals are already inside the
22.5deg tolerance at reset (success 0.762 there vs **0.500** on the 18 that must rotate) and
mean |dtheta| *increases* by 1.36deg over an episode. Build `contact_descriptors` on that.

### v33 RESULT — job 43679344, 18/18 cells, scored at digest `249434216cd2`

Six arms x 3 seeds, 600k steps. Each arm is v32's `curric` with ONE key changed. Goals >=3cm,
`logs/eval/v33/`, media in `media/v33/<arm>/`:

| arm | what changed | `model` | vs `ctl` |
|---|---|---|---|
| `ctl` | nothing (v32 `curric` rerun) | **0.674** | — |
| `widecone` | `push_cone_deg` 30 -> 90 | 0.604 | -0.069 |
| `freefinger` | `mask_inactive_finger=false` | 0.583 | -0.090 |
| `spread` | `object_theta_spread_deg=90` | 0.562 | -0.111 |
| `faceguard` | `guard_face=adjacent` | 0.562 | -0.111 |
| `midaction` | `finger_velocity` + `restrict_contact_actions` | 0.139 | **-0.535** |

**`ctl` reproduced v32 `curric` exactly (0.674 vs 0.674)** on a tree that had changed
`gym_env.py` and `contact_templates.py` — the replication check passed.

**Read these four ways.**

1. **No scaffold is free.** Nothing landed within 0.05 of `ctl`, so all four task-side
   scaffolds go in the fidelity-deviation list WITH their prices. Nothing gets quietly
   removed.
2. **The action space is the whole story, again.** `midaction` scored **0.139 — identical to
   v32's `raw` arm.** `restrict_contact_actions` on top of `finger_velocity` bought exactly
   **0.000**, curriculum or not. The contact FRAME does the work, not the no-retreat clamp
   (v18: 62% of contact breaks are tangential slides a normal-only clamp cannot touch).
   `midaction_s0`/`_s2` are the sweep's only cells with an empty distance bin, so they fail
   Bar 1 outright.
3. **THE FACE CONSTRAINT IS LEARNABLE, AT ~0.111.** `faceguard` was judged against 0.083
   (untrained under the guard), not `ctl`, and reached 0.562 — also beating the 0.483
   zero-shot number v32's best policy managed under `adjacent`. v31's strict form gave 0.000.
4. **What that does to v32's qualification.** The v32 headline was qualified because the arm
   ordering inverted under the face guard. `faceguard` does not erase that, but it moves the
   conclusion from "the curriculum may be exploiting a loophole" to "the loophole is worth
   ~0.111 and closes if you train with it closed." Honest statement: the curriculum helps,
   and a face-respecting push option is available at a known price.

### FACEGUARD ON ITS OWN DISTRIBUTION — training with the guard is COUNTERPRODUCTIVE

`logs/eval/v33_faceguard_own/`, digest `96762fdf1de4`, goals >=3cm:

| arm | trained with the guard? | `>=3cm` | `wrong_face` |
|---|---|---|---|
| `faceguard` | yes | 0.542 | 9.4% |
| `ctl` | **no** | **0.604** | 12.2% |

**The policy that never saw the constraint scores HIGHER under it (+0.062).** The guard cut
violations but not enough to pay for the success it cost — a terminating guard deletes
episodes that were on their way to arriving. Same mechanism as v31's 0.000, milder.

**So the reading from the tight benchmark alone was wrong, and this corrects it:**

- **The faithful face-respecting push number is 0.604** (`ctl` scored under
  `guard_face=adjacent`), costing **0.070** against the unguarded 0.674 — not 0.111.
- **Do not spend cells training under the guard.** `ctl` honours the constraint on ~88% of
  episodes without being told about it. **Enforce the face at EVAL and report the guarded
  score.** Cheaper and better.

**Still outstanding:** the same own-distribution scoring for `widecone` and `spread`. Given
this result, expect loosened-training arms to underperform `ctl` on the loosened task too.

### THE FACE-GUARD FINDING — it qualifies v32's headline

v32's two best policies, one key flipped (`logs/eval/v32_faceprobe/`):

| policy | unguarded | `guard_face=adjacent` | strict |
|---|---|---|---|
| `push_curric_s1` | 0.683 | **0.483** (wrong_face 21.7%) | 0.083 (81.7%) |
| `push_base_s1`   | 0.600 | **0.583** (wrong_face 15.0%) | 0.083 (86.7%) |

These policies leave the contacted face on 82-87% of episodes at a median of 12 ticks —
face switching is HOW they push, not an artifact. **The arm ordering inverts under the
guard.** "The curriculum helps on the task as scored" stands; "the curriculum learns a
better push option" does not. The `faceguard` arm is the test.

### v32 IS COMPLETE — job 43572361

12 cells, 600k steps, 2x2 of {curriculum on/off} x {restricted, raw actions} x 3 seeds. All
12 recorded one `GIT_DIFF_SHA=d93e35ff325287e6` and one launcher md5, so every cell ran the
same tree. Preregistered verdicts in `slurm/submit_sweep.sh`. Full findings in
`docs/PROGRESS.md` v32. Headlines:

- **v31's recontact Gamma arms were scored against a goal they could not reach.** `step`
  scores the ACTIVE finger against the goal's **L** slot. A state perfectly achieving the
  intended 6-D goal is scored arrived on only **254/500 (50.8%)** of resets. ~63 GPU-hours
  uninterpretable, and the old "raise the horizon" plan was tuning a task never scored.
- **Eq 15's curriculum, written literally, is INERT here.** Nested levels can only delete far
  starts; same-room median is 2.02/1.94/2.15/1.78cm across four levels against 2.00 with no
  curriculum. Replaced by a reverse curriculum (Florensa 2017 / Backplay 2018) — a deliberate,
  cited deviation.
- **`portal_arrival=true` puts the untrained floor at 0.271 on goals >=3cm** (0.42 in the
  crossing bins). A random policy crosses the doorway 42% of the time. Rejected for the
  benchmark; the doorway POSE is the proxy.
- **`portal_arrival` has never been enabled by ANY run**, so v31's headline audit fix was
  correct code the sweep never executed.
- **Final scores, goals >=3cm** (`logs/eval/v32_final/`, digest `249434216cd2`): `curric`
  **0.674** > `base` 0.583 > `raw` 0.139 > `curric_raw` 0.083, floor 0.042. The curriculum
  helps (3/3 seeds); it does NOT replace the action restriction. Task 11 died on a wandb init
  timeout and never trained.
- **WHILE A SWEEP RUNS, do not edit any file training imports** — a requeued cell re-reads
  the working tree.

### v31 RAN AND FAILED (2026-08-29) — superseded in part by v32 above

Two sweeps, both COMPLETED, 18 cells, ~126 GPU-hours. **The push ladder scored ZERO on all
nine cells; the Gamma-goal recontact arms scored ~zero on six of nine.** The one arm that
worked is the control.

```
job 42617855  push spec  1.2M, 8.0-9.4h/cell   final / best / last-25% mean (16-ep diag eval)
  spec          s0 0.000/0.062/0.002   s1 0.000/0.062/0.005   s2 0.000/0.000/0.000
  spec_raw      s0 0.000/0.062/0.001   s1 0.000/0.062/0.000   s2 0.000/0.000/0.000
  spec_settled  s0 0.000/0.062/0.003   s1 0.000/0.062/0.002   s2 0.062/0.125/0.007

job 42617867  recontact  1.0M, 6.3-7.1h/cell
  recon_base    s0 0.938/1.000/0.906   s1 1.000/1.000/0.941   s2 0.875/1.000/0.935  <- WORKS
  recon_goal    s0 0.000/0.062/0.003   s1 0.000/0.062/0.001   s2 0.000/0.125/0.005
  recon_full    s0 0.000/0.125/0.020   s1 0.000/0.188/0.051   s2 0.000/0.125/0.012
```

**The one positive result: `recon_base` is 0.906-0.941 (last-25% mean), 3/3 seeds, on
TODAY's code.** That is the rerun `docs/TODO.md` had asked for since v23; it confirms
recontact survived every interface change since, and it is the only v31 number comparable to
history.

**PUSH DIAGNOSIS — `wrong_face` starved training, but that is inferred, not measured.**
Tick-tracing `spec_s0`, 60 episodes:

```
terminations   wrong_face 43/60 (72%)  forbidden 9  contact_lost 4  horizon 3  arrived 1
median episode length 12 ticks    (66 ticks with the guard off)
counterfactual replay, SAME checkpoint:
  guard_face=TRUE   success 1/60   median closest approach 6.38cm
  guard_face=FALSE  success 2/60   median closest approach 4.84cm
  + no theta req    success 3/60   median closest approach 4.84cm
```

The face guard **terminated 72% of episodes at 12 ticks**, the most likely reason nothing
learned — but turning it off recovers almost nothing on an already-broken policy, so "the
guard broke TRAINING" is INFERRED. The clean test is one retrain with `guard_face=false`.

**Orientation was NOT the binding constraint, contrary to the preregistered worry.** 34/60
episodes reach |dtheta| <= 22.5deg at some tick; only **1/60** ever reach position < 0.4cm,
and dropping the orientation requirement entirely moves success 2/60 -> 3/60. This is
failure family 2 (never got close), not family 3 (arrives, no settle).

**RECONTACT-GAMMA DIAGNOSIS — no guard blowup; the goal is a conjunction that never fires.**
`recon_goal_s0`, 60 episodes: terminations are `horizon` 47, `object_disturbed` 13, so the
new `object_still` guard is NOT the problem. The 4-way conjunction is essentially never
satisfied: L within its 0.3cm anchor tolerance in **1/60**, R within 2.0cm in **2/60**, both
touch flags matching in **4/60**. Two fingertips must be placed inside a 100-tick horizon
sized for one.

**What this round establishes.** Every v31 change was individually justified by a
measurement, the gate grew 60 -> 132 checks, and the bundle still went to zero. The round
moved too many things at once for a 9-cell design to attribute — `hardmode`'s v29 mistake,
repeated at larger scale. **The next round must be single-factor off `lean`, not off `spec`.**

### v30 lean was CANCELLED mid-run (job 42569985)

Cancelled at ~3h55m, 570k-660k of 1.2M, all 6 cells. **The 400k snapshots survived on all
six**, so the budget-matched comparison against v29 — the primary question that sweep was
designed to answer — is still recoverable; the 1.2M endpoint is not. Last observed diag
eval: `lean` 0.688-0.750, `lean_raw` 0.000 on 3/3 seeds.

### Stage 1: recontact works (v23, reconfirmed v31). Push worked at v28, and does NOT at v31.

Every push number below is `eval_contact.py` on a fixed distance-stratified set; the printed
**env digest** must match before comparing two numbers, and each cell must be scored with
**its own action interface**. The v31 numbers above are the cheaper 16-episode diag eval on
each cell's own reset distribution — enough to read 0.000, not enough for an arm comparison.

### Recontact — solved, but not rerun since the action-space fix

`target_clip=goal_reward` took recontact 1/6 -> 6/6 seeds (job `41645529`, 1M). v23 on the
independent protocol (digest `9152a53a6b01`): `clip10_s0` **0.917**, `clip10_s3` 0.783,
diverged control 0.033. Critic gap +0.10 against **+44.8** pre-clip; finger-parking gone
(median final 0.30-0.36cm against `arrival_eps=0.4`). **Four clipped seeds unscored, and
all of it predates the srg pin** — check the protocol before quoting. Not rerun since v25's
interface change or v28's slip change.

### Push — the arc, compressed

| | what changed | same-room result |
|---|---|---|
| v21 | sampler drew face independently of goal; 56% of episodes needed a push AWAY | ~0.01 |
| v25 | `contact_frame` action interface — the reparameterization, not the clamp rate | 0.29 |
| v26 | slip derived from the friction cone (**a mistake, reverted in v28**) | — |
| v27 | free finger scored: `place` cosmetic, `unmask` harmful, **keep masking** | 0.21 |
| **v28** | **cone removed + 400k** | **0.739** |
| v29 | five scaffolds ablated; the action interface is the whole story | 0.822 best |
| v30 | goal diversity: cone 30 -> 90 -> 180 | 0.822 / 0.767 / 0.561 |
| v31 | ~8 changes bundled at once; `guard_face` strict killed it | 0.000 |
| **v32** | **reverse curriculum (band), pure sparse, pose goal** | **0.674** (>=3cm) |
| v33 | six one-factor scaffold removals, curriculum on | scoring |

Superseded but worth not re-deriving: `clamped` (v25) improved retention without improving
success, so **holding contact is necessary and not sufficient**. `place`
(`disengaged_away_deg=60`) removes `forbidden_contact` entirely (0/180) and is now adopted
in every cell; its apparent success gain was **+0.017 [-0.039,+0.078]** against the correct
transfer control — the eval that caught it is why the
RECONTACT GAMMA SCORING BUG item's pass (c) exists. `unmask` pushed `forbidden_contact` to 26.7-31.7% against a 19% prereg bar.

### v28 RESULT — push works (job 42248679, 400k, 3 seeds)

Final checkpoint, same-room benchmark, on the **same 60 episodes** v27 scored (verified
bit-identical; the digest moved `7b7d59c82155` -> `3ddae0eb3e93` only because
`push_range_min_cm` was added to the hash):

```
arm        0-3    3-6    6-9   9-12    12+    all   >=3cm
full      0.92   0.86   0.81   0.67   0.44   0.74   0.694
noclip    0.94   0.83   0.69   0.69   0.36   0.71   0.646
nomin     0.97   0.83   0.83   0.61   0.50   0.75   0.694
cone      0.83   0.42   0.08   0.06   0.03   0.28   0.146
cross0    0.86   0.53   0.22   0.08   0.00   0.34   0.208
cross50   0.86   0.72   0.53   0.61   0.33   0.61   0.549
v27 base  0.78   0.19   0.08   0.00   0.00   0.21   0.069
```

**The distance cliff is gone**; 12+cm went 0.00 -> 0.44, >=3cm went 0.069 -> 0.694. Seed sd
0.036-0.059, and `cone`'s range (0.250-0.333) never overlaps the working arms'
(0.633-0.800).

**Attribution is single-factor and lopsided.** Removing the flat slip law and restoring the
cone costs **-0.46**; the critic clip is worth **-0.03**; the 3cm goal floor **0.00**.

**The cone does not just cost performance, it CAPS the ceiling** — mean in-training eval by
step bucket: `full` 0.106 / 0.192 / 0.475 / 0.625 / 0.684, `cone` 0.073 / 0.125 / 0.199 /
0.277 / **0.256** (flat from 200k, then down). Budget 150k -> 400k bought the flat arm ~+0.45
and the cone arm ~+0.09. **That is why the 150k sweep read the cone's cost as 0.126** — a
fixed-budget comparison of unconverged runs ranks arms by convergence speed, not asymptote.

**Both preregistered predictions failed.** The 3cm floor did nothing (`full` and `nomin`
identical above 3cm, `nomin` better below). `cross0` did not fail — it reached 0.73 on its
own benchmark. That prediction reasoned from current success at 12+cm without checking
training DENSITY at 12+cm (8% for `full`, 100% for `cross0`).

**The failure mode inverted, and the critic is calibrated.** `full`: 74% arrived, 12%
horizon, 12% contact_lost, 2% forbidden, retention 0.92. **53% of failures come within 1cm**
(v27 `base`: 19%) — push now fails by arriving and not settling, which **reopens
`require_settled`**. Q-minus-realized went **+5.06 -> +0.10**, closing that open question:
the gap was a symptom of a policy that could not deliver its predicted value.

**Cross-room (32 episodes, goals 12-34cm), final / best:** `cross50` 0.667/0.729, `cross0`
0.615/0.729, `noclip` 0.604/0.490, `full` 0.531/0.458, `nomin` 0.542/0.438, `cone`
0.073/0.052. **Transfer is asymmetric and the varied task is the better teacher**: `full`
reaches 0.53 cross-room having never trained there, while `cross0` manages only 0.34 back on
same-room because it only learned to push +/-x (measured: 100% of its pushes, against `full`'s
34% north / 32% south / 18% east / 17% west). Best single cell anywhere is `full_s2` at
**0.844 cross-room**, beating the specialists on their own task.

### v29 PHASE 0 — three scaffolds costed at zero training cost

```
damping 6.00 -> 3.12   0.739 -> 0.706   MATCHED, identical 60 episodes (paired mean -0.033)
goal cone 30 -> 90deg  0.739 -> 0.722   different episodes, absolute
portal 20 -> 10cm      0.615 -> 0.198   3x COLLAPSE
```

**The two physics-fairness worries are not propping up the result. The GEOMETRY is.**

The damping value is still unphysical and that is derivable: `tau = mu*m*g*L`, and for a
body sliding on a plane `L` is the pressure-weighted mean radius of the contact patch. For a
10x6cm object, uniform pressure gives **3.12cm**, and **5.83cm (the half-diagonal) is the
hard ceiling** — all load on the two farthest corners. The code ships **6.00**, which no
pressure distribution can produce. Mirror of v26's mistake: that one over-derived, this one
under-derived.

**A claim that did not survive its own data.** "Rotation dominates every failure mode" is
false. Tick-tracing `full_s2` over 60 episodes: median max rotation **4.9deg**, p90 13.0deg,
9 of 60 ever exceed 10deg. The -81deg spin behind that story was the `angular_drag_arm_cm=1.0`
BUG, fixed long ago; the v21 steering-vs-retention tension belonged to the raw action space,
replaced in v25. Rotation is a non-issue here **largely because of** the fixed initial
heading and the 1.92x damping — which is why orientation goals cost so little today.

### The board is the live problem

```
room 25 x 30; usable object-centre box 13 x 18 (wall_margin 6); object 10 x 6
  usable width / object length = 1.3      max same-room goal = 22.2cm = 2.2 object lengths
portal 20cm of a 30cm wall = 67% OPEN, 2.0x object length, 5.0cm clearance each side
  objects spawn at y in [6,24] -- entirely INSIDE the gap [5,25]
  straight object->goal path blocked by the wall in 0 of 400 cross-room resets
```

Simultaneously **too small for long pushes** (12+cm same-room goals are 8% of episodes) and
**too open for crossing to mean anything**. "Crosses a room" currently means "a long straight
push that passes a doorway". A 10cm gap is exactly one object length — zero broadside
clearance — hence the 3x collapse above.

### v29 RAN AND IS SCORED (job 42300917, 24 cells, 400k)

**The handoff doc said NOT SUBMITTED and was wrong** — it was launched 2026-08-27 14:36 and
finished; the docs were written pre-submission and never updated. Scored 2026-08-28, 108
evals, three ways. Full numbers in `docs/PROGRESS.md` v29 RESULTS. Digest
`3ddae0eb3e93` -> `daee708c3fa6` from ADDED keys only; all 60 initial states verified
identical and `full_s0` re-scores at exactly 0.750, so v27/v28/v29 compare directly.

```
arm            all   >=3cm   vs full (paired)     arm          all   >=3cm
nogapassist  0.822   0.792   +0.083  p=0.017      narrowgap  0.317   0.174
physdamp     0.789   0.750   +0.050  p=0.233      rawact     0.217   0.090
unmask       0.756   0.701   +0.017  p=0.775      hardmode   0.183   0.028
full  (v28)  0.739   0.694        -                UNTRAINED 0.150   0.000
randtheta    0.694   0.639   -0.044  p=0.322      UNTRAINED  0.067   0.021  (finger_vel)
```

**THE FLOOR EXISTS** — untrained contact_frame is 0.150 all but **0.000 on goals >=3cm**,
scoring 0.75 in the 0-3cm bin alone. **Report >=3cm as the primary metric; the 5-bin mean
has a 0.150 floor.** `hardmode`'s 0.028 is the floor.

**Four of five scaffolds are free, and the action interface is the whole story.**
`nogapassist` is the largest single-factor effect in the round and it is **POSITIVE**
(+0.083, 3/3 seeds, p=0.017) — but `model_best` inverts the ordering, so it is PLAUSIBLE,
NOT CONFIRMED, and 400k does not settle it. `physdamp` holds under both checkpoints, so
**3.12cm should become the default**: the unphysical 6.00 was never buying performance.
`hardmode` (0.183) is confounded — `rawact` alone gives 0.217, so the collapse is entirely
the action interface, known since v25. Its preregistered 0.35-0.5 failed, but the arm cannot
answer its question.

**Pass (c) rescued an arm rather than killing one.** On its own task `randtheta` scores
0.717 while `full` transferred in scores 0.578, so a 90deg heading spread is a task the
BASELINE is worse at, not a harder task. `hardmode` scores 0.228 on its own task against
`full`'s 0.589 — beaten 2.6x by a policy that never trained there, which is a broken
learning setup, not a hard task.

**`bigroom` is NOT SCOREABLE on a shared benchmark, ever.** SB3's
`check_for_correct_spaces` includes the goal `Box` bounds `[board_w, board_h]`, so a 90x60
checkpoint raises `ValueError` against a 50x30 env. All 12 of its cross-board evals failed.
Board size can never be an arm in a sweep with a common benchmark without observation
surgery. Its standalone 0.354 also moves push distance (median goal 42.6 vs 21.7cm).

### Push's failure modes now split into three families

```
1 CANNOT HOLD CONTACT  contact_lost 66-69%, 25-50 ticks, retention 0.60   rawact, hardmode
2 RUNS OUT OF CLOCK    horizon 42-68%, 128-158 ticks, retention 0.90-0.98 narrowgap, cone,
                       median closest approach 3.0-3.3cm (11.6cm cross-room)  cross0
3 ARRIVES, NO SETTLE   42-69% of failures within 1cm, median closest 0.5-2.1cm  full,
                                                            nogapassist, physdamp, randtheta
```

**`require_settled` is live for family 3 ONLY** and is beside the point for 1 and 2.
`unmask` is a wash on success (p=0.775) and worse on safety (`forbidden_contact` 2% -> 8%
final, 11% best), so **keep masking** now stands on the safety column, not the success
column. The 8-11% is far below v27's 26.7-31.7% because every v29 cell trains with
`disengaged_away_deg=60`.

### Gates, and the repo state

**All green as of 2026-09-04:** `static` **38/38**, `geometry` 27/27, **`contact` 243/243**
(141 before Phase 0), `test_option_graph` 172/172, `fixture_eval` 18/18. The two new `static`
checks guard `score_sweep.cell_dirs` against the directory `finalize.sh` creates inside the
sweep it scores, and against a lexical sort putting task 10 before task 2. `contact` must live
in `test_code.py`, not under `tests/`, because `cmd_layering` forbids pymunk there.
**`ruff` is not installed in `tsmc`**, so `CLAUDE.md`'s lint gate cannot be run.

**THE TREE IS COMMITTED AND CLEAN, as of 2026-09-04.** `main` is at `4a1bfd9`. Phase 0, the
pre-launch audit and the v34 launch apparatus landed in four commits on 2026-09-04, verified
byte-identical to `GIT_DIFF_SHA=509bcc5e0fbe23b2` (the sha all 24 v34 cells recorded) in
every code file — only `docs/PROGRESS.md` and `docs/TODO.md` had moved since. **Sweep B is
the first sweep in eight to record `GIT_DIRTY=no` with a zero-byte `uncommitted.diff` on all
12 cells.** Keep it that way: do not edit the tree while a sweep is launching, because a
cell's `meta.txt` is written at task start and a doc edit mid-launch splits the record.

`tests/test_option_graph.py`, `tests/probe_edges.py` and `tests/summarize_horizon_sweep.py`
are gitignored by design (`tests/*`, exception only for `tests/fixture_eval.py`), so edits
there are real on disk and invisible to `git status`/`log`/`blame`. This file is gitignored
too.

**Not applied, deliberately:** `tools/prune_runs.py --apply` (123 files / 0.38 GB) and
`tools/prune_wandb.py --apply` (1 junk remote run). Both dry-run only.

**Where the CURRENT artifacts live.** v32: `logs/sweep_43572361/`,
`logs/eval/v32_final/` (scores), `logs/eval/v32_floor/` (untrained floor + `PROTOCOL.md`),
`logs/eval/v32_faceprobe/` (the face-guard measurement), `media/v32_best/` (4 arms x 6
clips). v33: `logs/sweep_43679344/` with its `PINS.txt`, scoring into
`logs/eval/v33/` and `media/v33/` (plus `logs/eval/v33_WRONG_PORTAL_do_not_use/`, kept
only as the record of the scoring bug). Older sweep artifacts are listed in
`docs/PROGRESS.md` under their own version headings.

## Next steps

**The authoritative list is `docs/TODO.md` ORDER OF WORK.** Short form, 2026-09-08 late:

1. **Score Sweep D by hand, per arm, against `PINS.<arm>.txt`** (job `45467480`,
   lands first). If `g_two_disp` returns 0.000 the next move is TASK DESIGN, not
   budget.
2. **Watch `eval/curriculum_level` at Sweep C's 600k rung** (job `45467495`). A
   stall there means the arms are not readable; the mitigation is designed and
   not built (`domains/contact/callbacks.py`).
3. **BUILD THE STAGE 1 LADDER — the critical path, needs no experiment, touches
   no frozen file.** `contact_hooks` is called by nothing; `MazeBundle`,
   `nav_entry_conditions` and `nav_descriptors` are all grid-native; no contact
   records exist on disk. 4-8 days. Build `contact_descriptors` on
   **ORIENTATION** — Sweep B collapsed the distance gradient to 0.083.
4. **Add a gate mode that asserts a flag changes an observable distribution.**
   The two longest-lived bug classes were silently inert or vacuously green.
5. **Decide `rich_obs`** — the twelfth interface key, still inside the digest.
   Moving it is correct and relabels every stored score. A decision, not a
   drive-by fix.

### Longer term, in order

1. **Phase B — composition. UNBLOCKED** since Bar 2 was met zero-shot (0.625).
   Reserved for the user's call. **The flat baseline and the composed task are
   the same piece of work**: on a single-edge task memo sec 5.2's flat arm IS the
   push option, so the comparison is empty until composition exists. The partial
   baselines on disk (v29 `rawact` 0.217, v32 `raw` 0.139, v33 `midaction` 0.139)
   price the ACTION SPACE, not the hierarchy.
2. **Stage 2 — domain randomization.** Friction and mass are fixed constants; a
   deliberate Stage 1 scope limit.
3. **Stage 0 residuals.** F1 (second budget rung, highest value), F2 (risk-aware
   routing), F2.5 (estimator ablations, no rollouts). **The cheapest path to
   publishable content, and none of it goes through push.**
   `docs/stage0_result.md` still does not exist.

**Closed, recorded so they are not reopened:** the Gamma scoring bug (verified
500/500); the face-centre spawn (P7); Bar 2 (0.625 zero-shot); the six untrained
floor protocols on disk; *"should `wrong_face` terminate"* — decided as
`guard_face=adjacent`, enforced at EVAL, never in training; and **the offset
door**, which shipped as board v2 (D4).

**SUPERSEDED 2026-09-08, and this one matters:** the 2026-09-04 entry read
"Gamma / Eq 13 is closed as posed (12 of 12 cells at 0.000) ... do not re-run
gamma at a larger budget." **That is no longer true.** Eq 13's positional form is
closed; contact-count Gamma (D1) plus the displacement guard (D2) reopened it,
and `g_one` scores **0.604** at 200k against a 0.208 floor. Sweep D is that
re-run. The "do not re-run on a flat zero" instruction now applies only to
`g_two_disp` returning 0.000.

## Stage 1 reference

```
board (push sweeps)  50 x 30 cm, one portal x=25 y in [5,25]   | default 80 x 60, no portals
object 10 x 6 cm, 0.20 kg   fingertip r=1.2cm   v_max 20 cm/s
physics 500 Hz   policy 25 Hz -> dt = 0.04 s   arrival_eps 0.4 cm
horizon  push 200 ticks | recontact 100 ticks   wall_margin 6.0 cm
reward   PURE SPARSE both templates: goal_reward=10 on arrival, every w_* = 0
         arrival TERMINATES, so Q* <= goal_reward = 10 EXACTLY (not 10/(1-gamma) = 1000)
clip     target_clip=10 (= goal_reward) is the recontact fix; sac_clipped.py; provable
         bound, not a tuned knob; active only for the first ~5k steps
guards   contact_lost after CONTACT_N_GRACE_STEPS=5 ticks -> 5*0.04*20 = 4.0 cm of finger
         travel allowed; rounding a 10x6 object needs ~7.5 cm, so REPOSITIONING IS
         FORBIDDEN inside one push option -- that is the recontact template's job
HER      SB3 `future` is INCLUSIVE of lag 0 (baselines uses t+1+offset, strictly future);
         min_progress_ticks=1 restores the reference convention
         HER positives are capped at arrival_eps + one tick of object motion (0.81 cm
         measured) -- ~3% of a 26 cm task, which is why sparse push could not learn
sampler  push_cone_deg=null keeps the historical face/goal-independent sampler (BUG
         INCLUDED, as a control); cone=30 gives median misalignment 12-15 deg.
         The goal is drawn on a RAY: pick a face -> push direction is that face's inward
         normal -> jitter +/-cone -> slide a random distance to the room edge. So the task
         is BUILT to be solvable by a roughly straight push, and cross-room additionally
         requires the ray to pass through the portal.
         push_range_min_cm (v28) floors that distance; short rays are REJECTED, not
         clamped, and the fallback redraws too (it leaked 3.7% at first). Cone-sampler
         FALLBACK rate (uniform room point, not on the ray) measured 2026-08-27:
         same-room 25.8%, cross-room 0%, narrow gap 0.2%, big room 0%.
         Training goal medians: srg=1.0 2.1cm | +3cm floor 4.8 | srg=0.5 14.2 | srg=0.0
         21.7. Benchmark median 7.2cm. The v28 floor closed most of that gap and bought
         **0.00** -- the mismatch was real but not binding.
eval     diag_eval_episodes=16 -> one episode = 6.25 pp, AND each cell evaluates on its
         own reset distribution -- measured to inflate success ~2x (v27). Use
         eval_contact.py's stratified set for any cross-cell claim; compare the printed
         env digest first. PROTOCOL PIN (v27): same_room_goal_prob=1.0, push_cone_deg=30,
         require_settled=false; push_range_min_cm=null pinned too (v28). v25 scored at
         srg=0.5, a HARDER set (mean goal 9.15 vs 7.59cm, 12+ bin max 29.3 vs 18.9cm), so
         its old numbers are not comparable. Score model_best AND model. REPORT SUCCESS ON
         GOALS >=3cm BESIDE THE 5-BIN MEAN (v28): 20% of the set is under 3cm where
         success needs <1cm of object motion (0.856 success under 1cm; median successful
         displacement 0.95cm), which halves every arm difference. base 0.150 -> 0.042 and
         legacy 0.294 -> 0.174 under the restriction. It is a free reweighting -- no
         re-run, no digest change.
action   action_interface=finger_velocity (default, bit-identical) | contact_frame
         (push only): (push, slide) in the contacted face's frame, re-derived every
         physics substep. Q* <= 10 holds under both.
slip     slip_model=speed_fraction slip_limit=1.0 (DEFAULT since v28): |v_t| =
         slide * v_max, so the only limits left are the clamp of the whole command to
         v_max and "no pulling". The finger may slide along a face with no push and
         pymunk's contact friction (mu=0.75, a native contact) decides what the object
         does -- friction is modelled ONCE, by the solver. friction_cone caps |v_t| at
         mu*push*v_max on top of that: a SECOND friction model, it forbids deliberate
         slip, and it FREEZES the finger at push=0. DEPRECATED, ablation arm only, NOT
         deleted (v26/v27 checkpoints must stay replayable). Measured cost at 400k,
         single-factor, same 60 episodes: 0.283 with the cone vs 0.739 without.
         Both models SCALE, never clip (no dead zone).
action space, after every filter (contact_frame): SAC emits 4 numbers in [-1,1]; the idle
         finger's 2 are zeroed; the active 2 become push = 0.5*(a+1) in [0,1] (affine, not
         clipped -- no dead zone) and slide in [-1,1]; re-derived EVERY physics substep;
         then gap_assist, then a clamp of the whole vector to v_max. What is left: any
         velocity up to v_max whose component into the face is non-negative -- the full
         half-disc, push and slide independent.
gap_assist true (default, INTERFACE key): forbids commanding retreat faster than the object
         recedes. Fires ONLY when the object recedes faster than the finger pushes, and
         then drags the finger inward to match it (a 0.1 push against a 12cm/s recession is
         applied as 12cm/s). A stationary object never triggers it. An ASSIST, not physics.
         finger_velocity has never had it, so false is the midpoint of full -> raw.
theta    object_theta_spread_deg=null (default): the object spawns at heading 0 in 300/300
         resets. Set it to a half-width and the face offset, the face normal AND the coned
         goal direction all rotate with it. Requires push_cone_deg. Measured rotation
         DURING an episode is tiny anyway -- median 4.9deg, p90 13.0deg -- because
         angular_drag_arm_cm=6.0 heavily damps it.
damping  angular_drag_arm_cm=6.0 is UNPHYSICAL. tau = mu*m*g*L, and L is the
         pressure-weighted mean radius of the contact patch: 3.12cm for a uniform 10x6,
         with 5.83cm (the half-diagonal) the hard ceiling for ANY pressure distribution.
         6.00 is 1.92x the standard value and above the ceiling. Costs only 0.033 (v29
         Phase 0, paired), so it is a correctness problem, not an inflation problem.
finger   mask_inactive_finger=true (default) ZEROES the free finger's action but leaves it
         servo-held in the object's path. false gives the policy both fingers with the
         Eq 40 guard as the only protection. disengaged_away_deg=null | half-angle (deg)
         of the spawn cone centred on the active face's outward normal; null is
         bit-identical (same RNG draw count and order).
board    50x30, one portal x=25 y in [5,25]. Room 25x30; usable object-centre box
         13x18 (wall_margin 6) = 1.3 object lengths wide; max same-room goal 22.2cm.
         Portal is 67% OPEN, 2.0x the object's length, 5cm clearance each side, and
         objects spawn at y in [6,24] -- INSIDE the gap. The wall blocked the straight
         path in 0 of 400 cross-room resets. TOO SMALL for long pushes, TOO OPEN for
         crossing to mean anything.
timing   push 150k = ~1h/cell, 400k = ~2.5-3h/cell, 1.2M = ~8-9.4h/cell (v31 measured)
         recontact 1M = ~6.3-7.1h/cell (v31 measured)
v31 obs  three CONCATENATED blocks [state][xi][goal-derived], matching Eq 18's
         pi(a | o(s), rho(g), xi). state 15, or 25 with rich_obs (+ contact normals 4,
         peak force 2, four nearest walls 4). xi 12 = template 2 + active finger 2 +
         contact face 4 + SOURCE interface class 4. tail 2 (2-D goal) | 4 (push pose) |
         6 (recontact Gamma). Totals: legacy push 17, push spec 41, recon_base 17,
         recon Gamma 43.
         The TARGET node is deliberately NOT in xi: Eq 18 puts it in rho(g), and HER
         rewrites rho(g) within an episode, so a target label in xi would disagree with the
         goal on ~80% of every relabeled batch (the v18 bug, in the block meant to be immune).
         CAVEAT: fingertip positions are object-RELATIVE but world-ORIENTED, while
         _wall_distances already ray-casts along the object's OWN axes. The observation is
         internally inconsistent and rotation equivariance must be learned from data.
         Fixing it must move the ACTION frame too and strands every checkpoint.
portal   a 10x6 object passes a 10cm gap at only 31.2% of orientations: the band is
         |theta| <= 28.1deg, worst-case y-extent 11.66cm at 59deg. portal_goal=true draws
         BOTH the goal pose and the object's start heading from that band -- edge
         FEASIBILITY (sec 6.4), a graph property, not something a policy can learn around.
her      her_valid_filter restricts relabel CANDIDATES to settled, guard-valid ticks.
         MEASURED on the spec config: 16.3% of ticks qualify, but 74.5% of future windows
         contain at least one -- 4.6x the retained signal of her_settled, which rejects the
         PAIR after drawing it. Empty window -> no relabel (keeps the real goal).
gamma    Gamma_l interface table in contact_templates.py: interface_targets (4/2/8 canonical
         placements) and sample_interface (continuous draw from the same class). Opposite
         face is face^1 -- NOT (face+2)%4, which maps +x to +y, an ADJACENT face, and
         silently made sec 6.3's "two opposing contacts" a corner grip. Pinch shares one
         along-face parameter so contacts are directly opposed; pivot's second contact is
         independent (that offset IS the moment arm). anchor tol 0.3cm, retract tol 2.0cm.
```

---

## The Stage 0 result

*Vocabulary: the console dump predates commit 4's rename and says `rung` where the code now
says `predictor`. Numbers unchanged, re-verified bit-identical post-rename.*

4000 stratified pairs, `option_budget=50`, `episode_budget=640`, `fixed_route`, frozen
`tests/fixtures/regions` weights (SAC, one seed, 90k). Route success **0.1797**.
290,213 transitions, 3m32s on one CPU.

```
 hops  grp    obs  noise     naive  marginal   handoff      R          rung      MAE    Brier   slope
    1   24  0.566  0.057    0.2247    0.1136    0.0810   1.40        naive   0.4497   0.3601   1.644
    2   28  0.161  0.033    0.5297    0.2553    0.0491   5.20     marginal   0.1924   0.1238   0.964
    3   16  0.000  0.009    0.6097    0.2002    0.0000    n/a      handoff   0.0461   0.0793   0.980
    4    4  0.000  0.002    0.5998    0.1934    0.0000    n/a
  all   72  0.251  0.034    0.4497    0.1924    0.0461   4.17     PASS: R >= 2.0, CI on D excludes 0
```

Coverage: naive/marginal/handoff all 72/72; chained 35/72 (1412 pairs), where handoff 0.0844
vs chained 0.0674. **Brier is a proper per-pair score independent of the grouping and orders
the predictors identically — the strongest single line of support.**

### The mechanism, confirmed

50 steps buys **5 cells**; region diameter is **8 cells**. Over 6,231 doorway legs, success
by entry-to-target distance: 0-2 cells 0.955 (n=491) · 2-4 0.860 (1729) · 4-5 0.662 (1650) ·
5-6 **0.047** (781) · 6-8 **0.000** (1580). The break falls exactly at the 5-cell reach and
`at_budget` tracks failure inversely across every edge. **Failing legs are legs that run out
of clock.** Necessary, not sufficient: success still declines within reach (0.955 -> 0.662)
because straight-line distance underestimates the Dubins path when the car must turn.

Three findings followed: all twelve interfaces are **directionally asymmetric** (1.6x-6.0x,
same doorway, same policy pair); the handoff advantage **grows with hop count** (R 1.40 at
h=1, 5.20 at h=2, because `predict_handoff` uses `p_bar_first` once and `H` exactly `hops`
times); and **long routes never reach their terminal leg** (at h=4, never more than 2 of 5),
which fully accounts for chained's 35/72 coverage. Terminal legs are not the bottleneck
(`v->goal` 0.828-0.917, except `5->goal` 0.638). Neither bug signature appeared: failures
spread across leg indices 0/1/2, and no high-calibration edge collapsed in composition.
Leg-0 failure is 31.6-35.4% in every hop stratum, as it must be if first legs are drawn from
one distribution.

---

## Audit findings (all fixed in commit 2; kept for the write-up)

Line-by-line read of `metrics.py`, `records.py`, `edge_model.py`, `calibrate.py`,
`fixture_eval.py`, plus arithmetic reconciliation against `metrics.json`. Vocabulary
predates commit 4's rename; see `docs/STRUCTURE.md` for the mapping.

1. **`D` was a bootstrap mean, not the point difference** (`d_point` 0.14629 vs `d_mean`
   0.13981). MAE is convex in `obs`, so resampling inflates every bootstrap MAE but only
   where `pred ~ obs`: handoff (~1 sd) by 0.0065, marginal (~5.7 sd) by ~0. **The same
   convexity biases `ratio_ci` low — report the two as one mechanism.** Bias runs toward the
   null, so the verdict is conservative. Recentered CI [0.13602, 0.15614].
2. **The h>=3 handoff zeros are real, not a missing-key default.** `n_groups` = `n_scored` =
   72, so every key resolved. `p_hat` was fit on 1,580 legs beyond 6 cells with **zero**
   successes; four such legs multiply to ~1e-5. The model is right; the measurement is
   floored.
3. **R is weighting-dependent:** group-equal (reported) 4.17 · pair-weighted ~5.86 · h<=2
   only, unfloored, **2.98**. Passes under all three. Defensible headline: *"R between 3.0
   and 5.9 depending on weighting, >= 2.0 throughout, with 2.98 on the measurable strata as
   the primary number."* Same caveat for handoff's slope (0.9797 includes ~20 free points at
   the origin).
4. **`p_bar` is the mean of `p_hat`, not a success count.** Empirical counts live in
   `beta_table` and feed the *planner*, not the ladder. Predictors 2/3/4 are **one estimator
   integrated against three measures** — design distribution, predecessor exit kernel,
   observed entries — so the 4x gap is attributable to distribution shift alone. Never
   describe `p_bar` as a frequency.
5. **Heading is uniform in every entry condition — a second mechanism.** `nav_strata`
   stratifies position and leaves heading uniform (deliberate), but composition exits leave
   in a narrow band along the approach normal. The marginal->handoff gap is **partly heading
   concentration, not only the 5-cell positional reach.** Say both.
6. **Two framing corrections.** (a) R 1.40 -> 5.20 is a *predictor-vs-predictor* advantage
   growing with depth — cite **H5 in Table 2**, not memo sec 5.5 (hierarchy vs flat).
   (b) `build_routes` does `if not ep.plan: continue` and a monolith episode has no plan, so
   every flat episode is silently skipped from the ladder. Correct, but
   `docs/stage0_result.md` must state it as a deliberate exclusion.

---

## Gotchas

- **TWO SAMPLERS, AND `curriculum_mode=band` PICKS THE ONE WITHOUT THE PORTAL CHECK.**
  `push_cone_deg` set does NOT mean `_sample_push_edge_coned` runs: if `curriculum_mode=band`
  (which every sweep since v32 pins) the call goes to `_sample_push_edge_reverse` instead
  (gym_env.py:851-853). The coned sampler verifies the object->goal ray passes the doorway
  (gym_env.py:727-733); the reverse one does not, and `portal_goal=true` hid that for eight
  sweeps by making the goal BE the doorway. Flipping `portal_goal=false` leaves 73.5% of
  crossing resets with no straight-line path on board v2 and 32.4% on board v1. **Check WHICH
  sampler a config selects before reasoning about its distribution**, and `curriculum_draws`
  counts rejection RETRIES in the reverse sampler, not fallbacks to a uniform draw — the two
  samplers use the same counter for different things.
- **A GUARD THAT GATES `_her_arrived` CAN SILENTLY TURN HER OFF.** `object_disturbed` is
  sticky and `~disturbed` gates the relabel arrival test (gym_env.py:1316-1321), so a guard
  firing early makes every later transition in the episode produce relabels that score as
  failures. Measured: firing at median tick 8 of 200 blacks out ~96% of an episode. The
  termination rate is the visible symptom; **the relabel-eligible fraction is the quantity
  that matters**, and `guard_terminates=false` does not restore it. Any new guard on a
  goal-independent predicate needs that fraction measured, not just its violation rate.
- **A `.zip` byte-compare is not a weight compare.** SB3 checkpoints embed timestamps, so
  `cmp` reports a difference at byte 11 on two bit-identical models. Compare the `.pth`
  members' tensors (`torch.load` each, max abs diff) — that is what "bit-identical floor"
  has to mean.

- **NEVER EDIT env/reward/eval CODE WHILE A SWEEP IS IN FLIGHT.** `finalize.sh` fires when a
  sweep's last task exits and runs `score_sweep.py -> eval_contact.py`, so an edit there
  scores the runs on different code than they trained under, and any `gym_env`/`physics`/
  `reward` change can move the digest and orphan every floor. Land deletions BETWEEN sweeps.
  Docs are always safe.
- **A CELL'S OWN DIAG EVAL IS NOT A CROSS-CELL NUMBER, and the CSV makes it easy to forget.**
  `eval/success_rate` in `progress.csv` is 32 episodes on that cell's OWN reset distribution.
  Two arms differing in a TASK key sit on different digests and their diag columns cannot be
  subtracted. Only `finalize.sh` on the pinned `PINS.txt` produces comparable numbers.
- **`progress.csv` ROWS ARE SPARSE — parse it with a carry-forward.** The CSV logger writes
  rollout rows and eval rows separately, so `eval/success_rate` and `time/total_timesteps`
  are almost never non-empty on the SAME row. A naive parser requiring both reports "no eval
  rows" for a fully-trained cell. Carry the last-seen timestep forward instead.
- **`set -e` PLUS A BARE `rc=$?` SILENTLY DISABLES THE FINALIZE BLOCK.** With `set -e` on, a
  failing training run exits the array task right there, skipping the last-task-standing
  block that moves the staging logs and submits `finalize.sh`. Use `python ... || rc=$?`.
  This is how 621 staging files accumulated.
- **A GUARD OUTCOME NOT NAMED IN `w_guard` FALLS BACK TO `w_m`, WHICH DENSE ARMS SET TO 0.**
  So a misspelled key is a silently FREE violation, not an error. `RewardWeights` now raises
  on any key outside `GUARD_OUTCOMES`; keep it that way.
- **HER CANNOT RECONSTRUCT A PER-EPISODE FACT.** `compute_reward` sees one transition, a
  stored info dict and a swapped goal. A once-per-episode credit reconstructed there pays
  EVERY qualifying row: `w_arrive_pos=3.0` measured 3.0/(1-0.99) = 300 of implied Q against
  `goal_reward=10`, on ~80% of every batch. Bounded drops are fine and listed in
  `reward.RELABEL_DROPPED`; unbounded asymmetries must be refused. **Prefer `w_prog`** —
  potential-based shaping is policy-invariant per goal and relabels exactly.
- **WHICH END OF `[0, target_clip]` A SHAPING TERM BREAKS DEPENDS ON ITS SIGN.** Negative-only
  shaping keeps `Q* <= goal_reward` so the upper clamp stays sound; only the lower clamp at 0
  is violated, which is what every recontact run has always done. POSITIVE shaping exceeds
  `goal_reward` and the clamp deletes the value the shaping exists to create. Conflating the
  two made `train_contact` refuse recontact's own archived baseline at startup.

- **`--pins` DID NOT FULLY DETERMINE THE SCORED TASK, AND MIGHT NOT AGAIN.**
  `tools/score_sweep.py` appended a hardcoded v29 `PORTALS` constant AFTER the pins; Hydra
  takes the last override, so `--pins` could not express the portal and v33's first scoring
  ran on the wrong board (36 of 60 benchmark episodes differed). Its help text said `--pins`
  "replaces TASK_PINS wholesale" and that was false. Now folded into `TASK_PINS`, with two
  `static` gates. **General rule: verify the DIGEST, and when a digest is unexpected compare
  the per-episode `d0` list, not the hash — a hash tells you something changed, the episodes
  tell you whether the task changed.**
- **TWO PIECES OF AUTOMATION WERE DEAD ON ARRIVAL IN ONE SESSION**, both because a tool's
  behaviour was assumed rather than checked: `squeue -o "%A_%a"` (see the finalize gotcha
  below) and `--pins`. Both were caught only by comparing output to a preregistered
  expectation. When wiring new automation onto existing code, run the existing code first.
- **A NEW ENV KWARG MOVES EVERY CONFIG'S DIGEST.** The env digest is `sha1` over every env
  kwarg except the six INTERFACE keys. So adding a kwarg — even one that defaults to the old
  behaviour — rehashes every config in the repo and orphans every stored floor and score. The
  face guard's `adjacent` mode was folded into the existing `guard_face` key for exactly this
  reason (`repr(False)` is unchanged), and `249434216cd2` was re-verified afterwards. **When a
  new option must be added, widen an existing key before adding a new one, and always verify
  the digest of a pinned protocol after the change.**
- **THE CONTACT-LOSS BUDGET DOES NOT BOUND FACE SWITCHING.** `CONTACT_N_GRACE_STEPS=5` is
  4.0cm of finger travel, but it only counts ticks spent NOT touching. A finger that keeps
  contact slides along the surface with no budget at all. Measured: v32's best policies leave
  the contacted face on 82-87% of episodes at a median of 12 ticks. The finger spawns at the
  face CENTRE of a 10x6 object, so the corner is 3.3cm away — ~4 ticks at 20cm/s. Any
  reasoning of the form "physics prevents X" needs the measurement, not the budget.
- **THE AUTO-FINALIZE HOOK WAS DEAD FOR THREE SWEEPS.** `slurm/submit_sweep.sh`'s
  last-task-standing test used `squeue -o "%A_%a"`, and on this Slurm **`%a` renders as the
  ACCOUNT name** (`43892866_hankyang_lab`), so the grep matched nothing, `still` never reached
  0, and the block never ran on v29, v32 or v33 — verified by three sweeps with no
  `slurm_logs/` and 621 orphaned files in `logs/slurm_staging/`. Now `-o "%i"` plus an atomic
  `mkdir .finalized` guard. **Still untested end to end; run `sbatch slurm/finalize.sh
  logs/sweep_<jobid> [tag]` by hand until a sweep proves it.**
- **PROGRESS IS `progress.csv`, NEVER `run.out` SIZE.** Python's stdout to a file is
  block-buffered, so a healthy cell shows an empty `run.out` for minutes. Two v33 arms looked
  stalled at zero log blocks and were fine.
- **VIDEO SELECTION BIASES WHAT YOU BELIEVE.** `eval_video_pick=auto` leads with the SHORTEST
  goals, which makes a mediocre policy look excellent — two rounds were spent looking at 1-2cm
  arrivals. `informative` is the default worth using: hardest arrivals plus the dominant
  failure mode. `tools/render_best.py` does this automatically per arm.

- **A FLAG THAT IS NEVER SET IS NOT A FEATURE, AND THE LAUNCHER IS THE ONLY PROOF.** v31's
  headline audit fix repaired a dropped `iface` in the arrival test. `portal_arrival` defaults
  false and no launcher has ever set it, so the fix was correct code the sweep never executed
  — and the open item it was supposed to close ("training never tests portal crossing") is
  still open. Same round: `her_valid_filter`, described as the change with the best measured
  payoff, was false in 6 of 9 push cells. **Before crediting a change, grep the LAUNCHED
  cell's `meta.txt` for the key, not the code for the fix.**

- **THE THING THE ENVIRONMENT SCORES AND THE THING YOU MEANT CAN BE DIFFERENT OBJECTS.**
  Recontact's Gamma arms: `step` scores `score_arrival(target=self._goal_xy[:2])`, which under
  a 6-D goal is always finger L's target, while `recontact_arrival` measures the ACTIVE
  finger, redrawn every reset. The correct 6-D test existed and was reachable only from
  `compute_reward`, i.e. only on relabeled samples. So rollout and relabeled rewards used two
  different definitions and 63 GPU-hours were uninterpretable. **The check that would have
  caught it in one line: construct the state that perfectly satisfies the goal and assert the
  env calls it a success.** It was scored arrived on 50.8% of resets. `gamma_goal` appears
  ZERO times in `test_code.py` — 132 checks, none end-to-end on the path that broke.

- **A DESIGN CAN BE FAITHFUL AND STILL INERT — MEASURE THE DISTRIBUTION, NOT THE CODE.**
  Memo Eq 15's curriculum is nested by construction (`I^(1) ⊂ I^(2) ⊂ ...`), so a level can
  only DELETE far starts; it can never make near ones commoner. With a sampler already bunched
  at a 2.0cm median, that changes nothing: measured 2.02/1.94/2.15/1.78 across four levels
  against 2.00 with no curriculum at all. The implementation was correct and the feature did
  nothing. **Before spending GPU on a task-design change, print the distribution it produces
  against the distribution with the feature off.** The literature is the check on the design:
  Florensa (2017) and Backplay (2018) both use MOVING WINDOWS, not nested sets, and ADR (2019)
  only expands because its sampler follows the boundary. Ours does not.

- **MAKING A TASK MORE FAITHFUL CAN DESTROY THE METRIC. FLOOR IT FIRST.** Scoring a crossing
  edge by "passed through the doorway" is exactly memo Eq 13, and it raises the UNTRAINED
  floor on goals >=3cm from 0.021 to **0.271** — 0.42 in the two bins where crossing goals
  live, because a random policy shoves the object through a 33%-open doorway whose straight
  path is blocked in 0 of 400 resets. **Regenerate the untrained floor for every protocol
  change, before the sweep. 4 cells, ~3 minutes** (`tools/make_board_v2_floor.sh`). A floor is
  specific to (interface, goal space, protocol) and none of them transfer.

- **THE CONTROL MUST SHARE EVERY MECHANISM EXCEPT THE ONE UNDER TEST.** A ramped curriculum
  arm compared against the OLD sampler would differ by two things. `curriculum_mode=band
  curriculum_levels=null` is the same sampler pinned at the full range, so the arms differ by
  the SCHEDULE alone. Costs one config value; buys attribution.

- **A CURRICULUM GATE CANNOT READ AN ENV THAT NEVER ADVANCES.** `eval_env` was built with
  `curriculum_levels` set and nothing advanced it, so it sat at level 0 forever and the 0.6
  threshold was reading the EASIEST distribution — which clears at once, and clears again at
  every level. Two envs: one pinned to the full task for reporting, one tracking the level for
  the gate. Also: the eval env is `Monitor`-wrapped, so reach `.unwrapped` to call into it.
  Both of these were found by SMOKE-TESTING THE REAL LAUNCHER, not by reading the diff.

- **A guard that TERMINATES converts a survivable behaviour into an instant failure, and
  the success rate hides which.** v31's `wrong_face` was correct on its own terms — the
  contact face is an edge parameter (Eq 7), so a finger that walks onto another face has
  left the edge. But v30 had already MEASURED that a finger sliding along a face rounds a
  corner and keeps contact, scoring 0.507-0.514 where a "behind the face is unreachable"
  derivation predicted 0.483. The guard killed exactly that behaviour: `wrong_face` fired on
  **72% of episodes**, cutting the median episode 66 -> 12 ticks, and all nine push cells
  trained to 0.000. **Before adding a terminating guard, check whether the behaviour it
  forbids is one an earlier measurement showed the policy USING.** And prefer a penalty
  first: `guard_terminates` is already per-env.

- **Eight individually-justified changes still make an unattributable bundle.** Every v31
  change had a measurement behind it and the gate grew 60 -> 132 checks. The sweep still
  went to zero on 9 of 9 push cells with no way to say which change did it — a 3-arm design
  cannot bisect 8 factors. This is `hardmode`'s v29 mistake at larger scale, and the v29
  entry for it is two screens above where v31 was written. **A justified change is not a
  free change. Count the factors against the arms BEFORE launching, not after.**

- **Both halves of "the goal got harder" need separate evidence, and the obvious half was
  wrong.** v31 added position AND orientation to push's arrival, and the preregistered worry
  was orientation (push rotates a 1.8deg median). Measured: 34/60 episodes hit
  |dtheta| <= 22.5deg at some tick, only **1/60** ever hit position < 0.4cm, and removing
  the orientation requirement entirely moved success 2/60 -> 3/60. **The conjunction failed
  on the term nobody was worried about.** Decompose a conjunctive arrival test into
  per-term hit rates before blaming a term.

- **A conjunction over N sub-goals gets exponentially rarer and HER cannot rescue it.**
  Recontact's Gamma goal scores four conditions at once (L within 0.3cm, R within 2.0cm,
  both touch flags). Measured over 60 episodes: 1/60, 2/60 and 4/60 respectively, joint
  ~0/60. HER relabels the whole 6-vector so relabeled goals ARE achieved by construction —
  which is exactly why the buffer looked healthy while the real task was unreachable. The
  single-finger 2-D version of the same task scores **0.94**. **Check the horizon too: it
  was 100 ticks, sized when ONE fingertip had to be placed.**

- **A verification probe is code too, and mine was wrong twice in one session.** Checking
  the v29 arms, I built `PlanarFingertipParams` without passing `board_w_cm`/`board_h_cm`,
  so three of four rows silently ran on the DEFAULT 80x60 board instead of the 50x30 the
  launcher sets. It made `narrowgap` look like it had a broken goal distribution (38.5cm
  median) when it is clean (22.5cm, 0.2% fallback). Separately, a retyped copy of the
  launcher's `case` block had a `same_room_goal_prob` typo the real file did not. **Drive
  the real file** (`eval "$(sed -n '/^ARMS=/,/^RUN_TAG=/p' launcher)"`) and **pin every
  parameter the launcher pins**, or you are verifying a different experiment.

- **A statistic that a threshold TERMINATES on cannot also measure failures against that
  threshold.** See the E3 entry below. Generalization: before quoting a number as evidence,
  ask what it *could* have read. If the answer is "only this", it is not evidence.

- **A physical constant with a closed form is not a tuning knob, and the check is cheap.**
  `angular_drag_arm_cm=6.0` shipped as "measured, not derived". For a body sliding on a
  plane the lever arm in `tau = mu*m*g*L` is the pressure-weighted mean radius of the
  contact patch — 3.12cm for a uniform 10x6, with the half-diagonal 5.83cm an absolute
  ceiling no pressure distribution can exceed. 6.00 is above the ceiling. Ten minutes of
  integration settles what a sweep cannot. Note this is the MIRROR of the friction-cone
  mistake: that one modelled friction twice, this one modelled it by hand where a formula
  existed. **Ask what the physics says BEFORE picking a number, in both directions.**

- **Replaying a frozen checkpoint costs minutes and reshapes sweeps.** v29 Phase 0 costed
  three scaffolds with zero gradient steps and two came back negative, cutting an arm before
  it ran and redirecting the round from physics to geometry. It works for any change the
  policy's OUTPUTS do not depend on (damping, sampler width, board geometry) and not for
  changes to the action space itself (unmasking, a different interface, the gap assist).
  Also note the limit: a board-size change rescales `obs()` (positions are divided by
  `max(board_w, board_h)`), so that one confounds transfer and must be trained.

- **Retiring a duplicate implementation can silently gut the test that guarded it.**
  `geometry.shortest_region_path` became a re-export of `planner.bfs_route` — correct, one
  implementation — but the gate golden-diffed the two against each other, so the check would
  have compared a function with itself and passed against anything. Replaced with an
  exhaustive simple-path oracle (validity + hop-optimality), a genuinely different
  algorithm. **When you delete one side of an equivalence test, the test dies with it.**

- **A renderer that ignores the task is worse than no renderer, and it looks fine.** The
  contact video drew board, object and fingertips for months with **no goal** — `Snapshot`
  simply had no goal field, so `plot_snapshot` could not draw one, and every mp4 was a
  valid file showing an object moving for no visible reason. Fixed 2026-08-27: the overlay
  (goal star, dashed arrival ring, object trail, red closest-approach ring, two-line
  caption) is passed to `to_snapshot`, because the goal is NOT in the state vector.
  Two encodings that matter and are easy to get wrong: fingertip **fill** is contact,
  fingertip **EDGE** is agency (heavy solid = driven, dashed = masked-and-servo-held, i.e.
  an obstacle); and `_GOAL_COLOR` must stay off tab10, since a blue star was exactly
  `tab10[0]` and vanished into finger L's own path. `arrival_eps=0.4cm` on a 50cm board is
  under 1% of frame width, so the ring has a floor of `0.02 * board_w` or it disappears.

- **`grep -r` in this repo LIES about `tests/`, and it cost two wrong deletion calls.**
  The shell's `grep` is a wrapper that passes `--ignore-files`, so it honours
  `.gitignore` — and `.gitignore` has `tests/*` (exception only for
  `tests/fixture_eval.py`). Every recursive grep silently skips
  `tests/test_option_graph.py`, `tests/probe_edges.py` and
  `tests/summarize_horizon_sweep.py`, which is where a third of the call sites live. On
  2026-08-27 this made `geometry.shortest_region_path` look dead when
  `test_option_graph.py` golden-diffs it against `planner.bfs_route` over every pair of
  every maze. **Use `command grep`, or walk the tree in Python, before calling anything
  unreferenced.** Same failure mode, opposite direction, as the note above about edits in
  `tests/` being invisible to `git status`.
- **A relative import defeats a naive import-graph scan.** The same sweep called
  `domains/nav/base.py` dead; `domains/nav/car.py` does `from .base import
  DynamicalSystem` and `DubinsCarSystem` inherits it. An AST walk must handle
  `ImportFrom.level > 0`, not just absolute modules. Corollary: an explicit
  `# noqa: F401` or a "re-exported, not defined here" comment is a deliberate façade —
  `nav/maze.py`'s `bilinear_sample` and `fixture_eval.py`'s `_pin_threads` both are.
- **A diagnostic can accept the data it is supposed to plot and silently ignore it.**
  `plots._draw_rollout_ax` took `X, goal, success, dist, midpoints` and drew only the
  region tint, so `eval_harness` wrote a `grid.png` captioned "worst 8 rollouts" with no
  rollouts in it — for months, because a valid PNG appeared every run. **Assert on the
  artists (`ax.lines`, `ax.collections`, the title), never on the file existing**, and
  check the assertions fail against the old body before trusting them. Second bug in this
  one file from having no coverage; the first is the `import pandas` note below.

- **An eval protocol's task keys must live next to the numbers, not in a launcher
  comment.** v25 scored at `same_room_goal_prob=0.5`; the v26 launcher's documented scoring
  command said `1.0`; nobody noticed, so the v25 SOTA checkpoint "replayed" at 0.4333
  against a stored 0.4167 and the digest silently moved. srg=1.0 is materially easier (mean
  goal 7.59 vs 9.15cm; 12+ bin max 18.9 vs 29.3cm), so every cross-version comparison was
  wrong until it was pinned. **The digest caught it and was ignored** — when a digest moves,
  find out WHY before scoring anything.
- **A digest mismatch has two very different causes and they need opposite responses.**
  Adding a key to the stamp (here `disengaged_away_deg`) changes the digest while the task
  stays bit-identical; changing a task key's VALUE changes the states. Tell them apart by
  regenerating the stratified set and diffing the initial states, not by reading the hash.
  Doing that is what vindicated the "null is bit-identical" claim and located the real cause.
- **`model_best.zip` is selected on the max of a 16-episode eval, so it is a lucky draw as
  often as a peak.** `unmask_place_s0`'s is from step 14,422 — 11% of its run — and
  `model_best` and `model` disagreed on the whole arm ordering of a 15-cell sweep. At 34s per
  eval, **score both**; if they disagree, the runs have not converged.
- **"Still rising at the budget cap" is a testable claim, and it was true for 15/15 cells.**
  Compare the last two eval buckets per cell and sign-test: 15/15 gives p ~ 3e-5. Worth doing
  before reading any arm comparison, because a fixed-budget comparison of unconverged runs
  ranks the arms by convergence speed, not by asymptote. The tell here was `ep_len_mean`
  going 11-14 -> 78-129 ticks: the same step budget bought ~8x fewer episodes.
- **A cross-group arm needs its own transfer control or it will manufacture a win.** `place`
  read +0.056 against `base` in its own group and **+0.017 [-0.039,+0.078]** against the same
  `base` checkpoints scored under `place`'s reset. The naive read would have adopted a
  reset-distribution change as a policy improvement — memo sec 7's named failure mode, caught
  by one extra eval per seed.
- **Identical success rates across seeds are not evidence of duplicated cells — check the
  MEMBERSHIP.** `base`'s three seeds all scored exactly 9/60, the v20 duplicate signature,
  but their success sets differ by 4-6 episodes and mean |dQ| is 0.9-1.4. They also all
  succeed on the same low-index episodes, which is the aiming result restated: solvability is
  set by the task, not the seed.
- **The login node has `nproc=1` and sits at load ~40.** 36 evals there oversubscribe one
  core and are bad citizenship; `slurm/score_sweep.sh` runs the same matrix on `shared` with
  12 cores in ~4 minutes. Check `nproc` before parallelizing anything locally.
- **`portals=[{...}]` bites in heredocs too, not just interactive shells.** Bash
  brace-expanded it into three words inside an sbatch script and Hydra received
  `portals=[x:25.0]`; the fix is the same shell variable. `tools/score_sweep.py` passes it as
  a Python list element and is immune — prefer that path.

- **SB3's `HerReplayBuffer` relabels `desired_goal` but never touches `observation`**
  (read `_get_virtual_samples`, 2.9.0). Any goal-derived feature baked into `"observation"`
  goes stale on ~80% of every batch (`her_ratio` at `n_sampled_goal=4`). Fixed two ways in
  v18: recompute at relabel time via a custom subclass (push), or put the goal in a frame
  invariant to which tick the relabel drew from (recontact's object-frame goal). **Before
  adding any field to a goal-conditioned `"observation"`, ask whether it depends on the
  current goal.**
- **SB3's `HerReplayBuffer` never recomputes a relabeled transition's `done` from its new
  reward** — it keeps the original rollout's flag (v19). A virtual goal that scores the
  arrival bonus still lets the critic bootstrap past it. Patch `dones` in the subclass by
  OR-ing in whatever boolean `compute_reward` uses. Only push has a subclass
  (`her_buffer.py`).
- **A storage tool must measure `st_blocks`, not `st_size`.** wandb's `.wandb` files are
  sparse: 1127 MB apparent against 351 MB actually on disk, a 3x overstatement. `du`
  reports blocks and is right for "will the disk fill up"; `os.path.getsize` reports
  apparent size and is right for almost nothing here. Cross-check any size report against
  `du` before acting on it.
- **wandb was never the storage problem.** Measured 2026-08-26: remote is 155 runs /
  0.25 GB against a 100 GB tier; local `logs/` is 1.2 GB of which **1.1 GB is 290 SB3
  checkpoints**. Growth is ~104 MB per 16-cell sweep. **Old checkpoints are load-bearing**
  — v23's transfer-vs-retrained result needed job 41613939's checkpoints from two days
  earlier, so a date-cutoff purge would have destroyed an experiment before it was run.
  Prune on evidence (`max(eval/success_rate) == 0.0` makes a `model_best.zip` provably an
  early snapshot), never on age.
- **The incident behind the rule above.** v25's first scoring pass forwarded only task
  overrides, so `contact_frame` policies emitting (push, slide) were read as (vx, vy). The
  table looked like a decisive negative and was pure artifact — the v16 failure mode again.
  Caught only by replaying one checkpoint on the same seeds and getting a different
  termination histogram than the stored json. **Do that replay check whenever a stored
  result surprises you.**
- **A dominant failure code can be an artifact of the setup, not the policy.** Masking the
  inactive finger does not remove it — the velocity servo *holds it in place*, so once
  `contact_frame` made pushes long enough to travel, the object drove into our own parked
  fingertip and the Eq 40 guard called it `forbidden_contact` (8% -> 17-19%). The tell was
  **timing**: it fired at tick 26-31 instead of 8-9, which is arrival at an obstacle, not a
  bad grasp at reset. **Check WHEN a failure fires, not just how often.**
- **Before a training sweep, ask whether a frozen policy can already answer the question.**
  Replaying the SOTA checkpoint under only a changed reset cut `forbidden_contact` 13.3% ->
  1.7% at zero training cost, confirming the mechanism before 15 cells were spent. The
  limit: this only works for changes the policy's outputs do not depend on. Replaying a
  masked checkpoint *unmasked* is meaningless — it animates two outputs it never learned to
  control, so `mask_inactive_finger` had to go in the sweep.
- **"Ran out of time" and "never got close" produce the same success rate and need opposite
  fixes.** Push's failures looked like a horizon/settling problem (47-49% `horizon`, median
  final distance 4.7-6.7cm). Recording **closest approach** showed a median of 3.09cm and
  only 1.40cm given up after it — never-got-close, so a 12-cell `horizon x require_settled`
  sweep died before it ran. **Record the minimum of a distance over an episode, not only its
  final value.** Cost: one hypot per step. **But watch for a tautology in the same
  statistic:** the accompanying "0% of failures ever reached `arrival_eps`" is 0% BY
  CONSTRUCTION, because arrival terminates at that radius. A threshold that ends the episode
  cannot also be a threshold failures are measured against. Ask what a statistic *could* have
  read before quoting it as evidence. **And decompose the miss** — the start/end/goal
  triangle is free from data already on disk, and it showed the shortfall is about half the
  error, not a footnote to aiming.
- **Do not model the same physics twice — check whether the SOLVER already does it.**
  v26 replaced a tuned `slip_limit` with `mu * push * v_max`, reasoning that a derived
  constant beats a swept one. The derivation was fine; the premise was not. The finger is a
  FORCE servo and its contact with the object is a native pymunk contact at mu=0.75, so
  **pymunk already resolves slip** — commanding a tangential motion friction cannot support
  makes the finger slide, which is what a real finger does. Capping the COMMAND at the
  friction cone on top of that forbids deliberate slip and freezes the finger at push=0, so
  it cannot slide along a face without also driving the object. Measured cost on goals
  >=3cm, 14 cells, one eval set: cone 0.042 < flat 0.5 0.109 < flat 1.0 0.170, monotone.
  Reverted to a flat cap in v28; `friction_cone` kept as an ablation arm. The surviving
  half of the v26 lesson: when superseding a constant, **keep the old path bit-exact** —
  archived checkpoints must stay replayable under the interface they were trained on.
- **The interface/task key split has a general form.** A key is an INTERFACE key if it
  changes what the policy's outputs *mean* (`action_interface`, `slip_model`, `slip_limit`,
  `restrict_contact_actions`, `mask_inactive_finger`) — take it from each cell's `meta.txt`
  and exclude it from the digest. It is a TASK key if it changes what success *is* or which
  states are visited (`require_settled`, horizon, sampler, `disengaged_away_deg`) — pin it
  at one protocol value for every arm and keep it inside the digest. **Two arms differing in
  a task key are two experiments, not two arms**, and need a transfer eval to be compared.
- **A reset-distribution change forces a re-baseline and engages memo sec 5.2.** Anything
  touching how episodes start (cone sampler, spawn placement) changes the env digest, so
  every prior number is incomparable, AND the flat baseline must inherit the identical
  distribution or "hierarchy wins" is a curriculum artifact (memo sec 7's named failure).
- **Anything sampled at `reset` must be re-read every episode.** `eval_contact.py` read the
  active finger once per checkpoint; `reset` re-samples it, so contact retention was
  counted against the wrong finger on most episodes and v23's retention numbers were wrong
  by ~2x. Success, termination and Q were unaffected because they never used it — which is
  why it survived a whole iteration unnoticed. **A per-episode quantity cached outside the
  episode loop is a silent corruption, not a crash.**
- **A per-cell eval distribution can manufacture an effect that isn't there.** v22 read
  `cone=30 srg=1.0` at 0.340 against a 0.000 control and called the sampler fixed. Scored
  on ONE common stratified set, the retrained policies land at 0.142 against transferred
  policies' 0.108 — inside seed noise. The original gap was 11-of-16 eval episodes under
  3cm (median 1.9cm) versus 3-of-16 (median 15.6cm). **Two success rates measured on
  different reset distributions are not comparable, however carefully each was computed.**
  `eval_contact.py` exists for this and prints an env digest; check it matches.
- **Low critic loss is not evidence of a calibrated critic.** A critic can converge to a
  stable overestimate — the TD residuals agree with each other and stay small. Push's
  `critic_loss` of 15-36 said nothing; the Q-vs-realized comparison (+1.62/+1.97, max 8.90
  against the provable bound of 10) is what showed it was fine. Measure the gap, not the
  loss.
- **A regression test that never triggers the branch it names is worse than no test.** The
  first version of the `restrict_contact_actions` check drove a scripted action that never
  opened the contact gap, so the clamp never fired and the test would have passed against
  a completely broken refactor. Replaced by an equivalence test against the pre-refactor
  math over 400 random states, **with an assertion that the clamp actually fired on >100
  of them.** Assert the branch was exercised, not just that the output matched.
- **A critic-target clip can be inactive for 99.5% of training and still decide the run.**
  `target_clip_frac` was ~0.97 for the first ~5k steps and ~0 thereafter, yet it is the
  difference between 1/6 and 6/6 seeds converging. **Judge an intervention by when it is
  active, not by its time-average** — a near-zero mean clip rate is not evidence the clip
  did nothing.
- **The HER target-clipping convention needs its constant re-derived when success
  TERMINATES.** OpenAI baselines uses `clip_return = 1/(1-gamma)` — correct for Fetch,
  where success does not end the episode. Here arrival terminates and every other weight is
  zero, so `Q* <= goal_reward` **exactly**: 10, not 1000. A 100x difference that decides
  whether the clip fires at all (worst observed Q was 539, so the conventional bound would
  have been a no-op). **Clipping is not reward shaping** — it constrains the critic's
  regression target, so pure-sparse stays pure-sparse.
- **HER cannot manufacture signal for a task whose object barely moves, and no threshold
  fixes it.** `arrived` compares the relabeled goal to `ag_{t+1}` ~ `ag_t`, so every
  positive HER example is capped at `arrival_eps` + one tick of object motion — measured
  0.81–0.86cm on a 26cm task (v20). **Check this ceiling before tuning any HER gate:** if
  it is small relative to the task, the problem is exploration, not credit assignment.
- **`min_progress_cm` must stay strictly below `arrival_eps`, or HER goes silent.** Both
  gate conditions hold only if the object moves more than `min_progress_cm - arrival_eps`
  **in the single relabeled tick**; median per-tick displacement is 0.0098cm, so 0.5 vs 0.4
  keeps 0.33–1.0% of pairs and >=0.9 keeps provably zero (v20). Worse, survivors are the
  fastest-moving ticks (21–43x median), which under `require_settled=true` can never be
  real successes — a too-high threshold inverts the signal, not just thins it. **Prefer a
  temporal gate (goal drawn >= N ticks ahead), which is invariant to object speed.**
- **A HER anti-degenerate filter must compare against the transition's OWN pre-action
  position, not the episode's start.** v18 compared to `info["start_achieved_goal"]`, but
  HER relabels per-pair, so it could not distinguish an informative early-tick pair from a
  trivial late-tick one in the same episode (v19, user-identified). Fixed by reading
  `info["pre_achieved_goal"]`. The per-pair version provably subsumes the per-episode case,
  so there is no accuracy tradeoff.
- **Sample an option's initiation set and target set CONSISTENTLY.** Push drew the contact
  face independently of the goal, so the median episode needed the object pushed ~98 deg
  away from the only pushable direction, 56% needed a push that moves it AWAY, and ~5% were
  solvable by a straight push (v21). Memo Eq 7's `xi_e` contains the **object face**, and
  Eq 12 puts the finger on the west face for an eastward push. **Whichever of (face, goal)
  is free should be chosen to match the other.**
- **An aggregate success rate can hide a fully learned skill when most episodes are
  unsolvable.** Push scored 1–2% and was indistinguishable from doing nothing (Fisher
  p=0.62) — the SAME checkpoints score 34–39% on the corrected sampler with zero
  retraining. **Before concluding a policy learned nothing, establish what a trivial
  baseline scores on the same distribution.** Corollary, from the v22 sweep: once the task
  is fixed, check that *retraining* actually beats the zero-retraining transfer — a fixed
  task and a better policy are different claims.
- **A guard can forbid the very maneuver the task distribution requires.** `push_guard`
  fires `contact_lost` after 5 ticks = 4.0cm of finger travel, while walking around a
  10x6cm object is ~7.5cm — off by ~2x, so ~94% of episodes were push->recontact->push
  tasks being trained as one edge (v21). **When one template's failures dominate, check
  whether the fix belongs to a DIFFERENT template.**
- **Steering and contact retention are mutually exclusive under a raw finger-velocity
  action space.** Raising the goal-directed weight drops retention 100% -> 42% and
  displacement to ~0 (v21): steering needs tangential motion, which slides the finger off
  the face. `arctan(finger_friction) = 36.9deg` is the right order of magnitude but not the
  whole answer — measured deviation reaches 65–78deg at p90 while contact holds, because the
  object rotates and pymunk's solver is more permissive than a Coulomb contact.
  **Superseded for the ACTION LAYER in v25 (contact_frame), and the v26 friction-cone
  follow-up was REVERTED in v28:** "we should not command motions a real finger could not
  make" was wrong — a real finger makes them and slides, and pymunk models that already. The 65-78deg observation still stands as a fact
  about the simulator, and memo sec 6.4's "measure it" advice still governs *geometric*
  parameters like `I_e`'s proxy — just not this one.
- **A reachable set does not need Hamilton-Jacobi analysis in this architecture.** `p_hat`
  (Eq 22) IS a learned, calibrated, state-dependent reachability probability, and the
  planner consumes `-log p` (Eq 31/32), so a hard 0/1 boundary would have to be softened
  again. **Use a cheap conservative geometric proxy for `I_e`, pick its one parameter by
  measurement, and let `p_hat` learn the real thing inside it.** `A(v,e)` (Eq 10) detects a
  too-loose proxy: high `Var[p_hat]` within a node.
- **A correlation ranking over training logs does not identify a mechanism**, even a strong
  and consistent one. `ent_coef` was recontact's strongest anti-correlate of success
  (-0.63/-0.72), reading as re-injected noise — but measured action std *fell*, and eval
  runs `deterministic=True` anyway (v20). The tick-trace caught it.
- **`model_best.zip` is meaningless when `eval/success_rate` never exceeds 0.0.**
  `ContactPeriodicEvalCallback` saves on `> best` initialised to -1.0, so the first eval
  always writes and nothing beats it. Check the metric actually moved.
- **A swept bash variable can be computed correctly and never reach the command.** v16's
  push branch computed `WM=${WMS[...]}` but never put `w_m=${WM}` in `EXTRA_OVERRIDE` — 8
  "different" cells trained at one value, with no error. **Verify a sweep varied what it
  claims to by grepping the *logged* command** (`meta.txt`'s `EXTRA_OVERRIDE=`), not the
  launcher script.
- **A resubmitted array can silently duplicate cells that did not change.** Job `40944664`
  cells 4–7 are bit-identical to `40910275` cells 2–5 in every learning metric — 24
  CPU-hours for nothing (v20). File hashes differ (`time/fps`), so compare learning
  columns. Corollary: the pipeline is deterministic, so any curve is exactly reproducible.
- **`meta.txt`'s `GIT_COMMIT` cannot distinguish a pre-fix run from a post-fix one when the
  fix is uncommitted.** Jobs `40944664` and `40957220` both record `a7c153a`. Both sweep
  scripts now record `GIT_DIRTY` and `GIT_DIFF_SHA` and save `uncommitted.diff`, but
  **commit before submitting** or provenance is a guess.
- **`portals=[{x:...,y_lo:...,y_hi:...}]` must reach Hydra through a shell variable.** Bash
  brace-expands `[{a,b,c}]` into three words; assignment suppresses it. Not a code bug.
- **Check `docs/STRUCTURE.md`'s Environment section before concluding the env is missing.**
  `module load python` then `mamba activate tsmc` — the env is `tsmc`, not `to-smc` as
  `environment.yml`'s `name:` says (pre-existing mismatch).
- **Never pipe `mamba activate`'s own line through another command** — the pipe runs it in
  a subshell and the exported vars do not persist; `python` then silently resolves to base
  Miniforge. Symptom: `sys.executable` under `/n/sw/Miniforge3-.../bin/python` and
  `ModuleNotFoundError` for `hydra`/`pymunk`. Run it as its own unpiped line.
- **`hydra-core==1.3.2`/`omegaconf==2.3.0`/`antlr4-python3-runtime==4.9.3` are declared in
  `requirements.txt` but were not installed into `tsmc` until v14** — declaring a dep does
  not install it into an existing conda env. Check this before assuming the migration
  broke (only after confirming the activate took effect).
- **`WANDB_API_KEY` lives in `~/.bashrc`, not `~/.netrc`.** A plain non-login shell (this
  agent's `Bash` tool) fails `wandb.Api()` with "No API key configured" even though every
  `sbatch` job works. `source ~/.bashrc` before any ad-hoc wandb call. Never log video,
  images, or large artifacts — local disk only.
- **Two processes sharing one `WANDB_RUN_ID` can race at `wandb.init()` itself** (v15) —
  wandb's backend 409s on the duplicate and the losing process's entire `history` silently
  never lands, with no Python-side exception. Confirm via `run.history()`/`run.summary`,
  not by eyeballing. Fixed by staggering `init()` per `SLURM_ARRAY_TASK_ID`. wandb runs are
  also not crash-safe (no `try/finally`) — cosmetic only, left as-is.
- **`EpisodeRecord.from_dict()` silently drops unknown keys** (STRUCTURE.md's THE TRAP).
  Renaming a wire field invalidates every record on disk with no error.
- **`calibrate.py`'s `option_budget` defaults to the FROZEN horizon** — 200 in the fixture
  weights, not 50. Omitting it calibrates at a different clock than the observation and the
  model json looks normal. **Pass it explicitly**; the `_h50` artifact names are the only
  record of which clock was used.
- **An oracle predictor must never gate a confirmatory sample** (D1). Generalize: *any
  statistic whose definedness depends on the outcome will select on the outcome.* This
  dropped 37 of 72 routes, keeping the short high-success ones, before it was caught.
- **A normal-approximation noise floor returns exactly 0 at p=0.** Use Jeffreys.
- **Naive's source is `logs/probe/edges_n8_h50.json` -> `regions[v]["mixed"]["rate"]`** at
  the matched clock. `summary.json`'s per-region block is at the frozen horizon (MAE 0.6094
  vs 0.4497) — a labelled secondary variant only. `mixed`, not `terminal`.
- **A handoff doc's "confirmed on disk" is a claim, not a fact — verify by reading the
  file.** v12 said the `RewardWeights` ablation was on disk; it wasn't (v13). One `Read`
  call versus a wasted retrain. **This applies to future sessions reading *this* doc.**
- **Background training jobs compete with gate runs for CPU.** `fixture_eval` took >2
  minutes while two training runs shared the node — contention, not failure.
- **`train.py`'s plotting path had zero test coverage, and it showed** —
  `option_graph/analysis/plots.py` was missing `import pandas as pd` and referenced an
  undefined `region_csvs`, found only by smoke-testing a real run. **Smoke-test new
  diagnostics with real (even tiny) data**; an empty CSV takes the early-exit branch.
- **Never `set -u` in an sbatch script that sources `~/.bashrc`** — FASRC's `/etc/bashrc`
  reads `$BASHRCSOURCED` with no default and the shell dies. **FASRC CPU partition is
  `shared`**; `gpu`/`gpu_test` for training; `sbatch` requires `--time`.
- **`mkdir -p logs/eval` before any `run_eval | tee`** — `tee` opens its file before the
  process writes, so the run dies with no output and no explanation.
- **A new module under `option_graph/` must import `eval_harness` lazily**, inside the
  function body — `eval_harness` is in `LAYERING_EXEMPT`, a new sibling is not.
  Serialization goes through `records.json_safe` — the single definition, guarded by
  `static`. Do not add a local copy: the five that existed diverged on exactly the
  `np.ndarray` branch, and only the ones that had it could serialize a `PHat`.
- **Do not transfer files as base64 through a chat client** — it inserts zero-width
  characters and re-wraps lines. Plain-source heredocs with a `wc -l` check work.
- Misc invariants: `gate` and `mode` have no defaults in `ExecConfig`, deliberately
  (`fixed_route` for the predictor comparison; `replan` never). `hops` must come from the
  eval pair for BOTH arms or the monolith scores 0 on every pair. Terminal legs have no
  exit line, so `A(v,e)`'s tangential term is nan (expect 24/33 coverage). `p_hat` predicts
  crossing (`reached_position`); `reached_interface` is a diagnostic, never a fit target.
  `edge_model` and `calibrate` are numpy-only — do not reintroduce torch. `num_pairs`
  changes the per-hop quota and rng call order, so pair sets are not nested.
  `sample_eval_pairs` samples with replacement but `draw()` redraws jitter and heading, so
  no dedup is needed. `failed_option()` labels a goal-cut-short leg `"goal"`, so failing-
  index counts can exceed failure counts (457 vs 442 at h=1; `n_cut_short` = 18, 457-442 =
  15).

---

## Physics reference

```
v0 = 1.000   omega_max = 8.0   dt = 0.1   cell_size = 1.0
r_min = v0/omega_max = 0.125
steps_per_cell = cell_size/(v0*dt) = 10        per-step travel = 0.1
REACH AT BUDGET 50 = 5 CELLS   region diameter = 8 cells   <- explains everything at h>=3
FULL TRAVERSE NEEDS >= 80 STEPS                <- the F1 feasibility floor
wall_margin 0.25   arrival_eps 0.4   line_offset_cells 0.5   alpha 45 deg (half-angle)
option_budget 50 (experimental)   episode_budget 640 (fairness anchor)
nine_rooms: h_region 160 (derived), region_gamma 0.99375  | flat 640, flat_gamma 0.9984375
giant:      h_region 280, region_gamma 0.996428571        | flat 840, flat_gamma 0.998809524
nine_rooms  9 regions, 12 interfaces, 24 directed edges, 33 legs, hop strata 1-4, diameter 8
            degrees: 4 corners deg 2, 4 edge-rooms deg 3, center deg 4
            125 = Sum((deg+1)^2)   68 = Sum(deg^2)   10,100 = 12,500 - Sum(deg)*100
giant       8 regions, 10 interfaces, 20 directed edges, hop diameter 3, diameters 4-14
calibration 125 conditions x 100 trials = 12,500 rollouts = 420,352 env steps, 0 gradient steps
observation 4000 pairs, 72 routes (24/28/16/4 by hops), 6,231 doorway legs, 290,213 env steps
budget      training 90,000 both arms | calibration 420,352 vs 0 | N_total 510,352 vs 90,000
```

`h_region = 160` is the derived per-region clock; `option_budget = 50` is an experimental
choice layered on top of it. The 5-cell reach that follows is the central fact behind the
floored strata, the directional asymmetry, and F1. **Record both numbers, always, in every
domain.**
