# Progress log

Dated: question -> what ran -> result -> next. Newest last. Session labels (v13-v19) are
the handoff-doc versions they came from; dates are from the commits they landed in, so a
session spanning several days carries the date of its commit.

Standing rule behind every entry below: **a training-time CSV or wandb dashboard is not a
diagnosis.** Every mechanism recorded here came from tick-tracing a saved checkpoint —
`SAC.load(path, env=env)` (HER models need `env=`), step deterministically, read
`env._x` / `IDX_CONTACT` / `IDX_NO_CONTACT_STEPS` / `IDX_OBJ_VEL`. Numbers that had not
yet been tick-traced when written are flagged as such.

---

## Stage 0 — closed (through 2026-08-08)

The preregistered question was answered, the v9 audit's three labelling fixes landed, and
the full vocabulary rename completed. Result numbers were re-verified bit-identical
against the pre-refactor `metrics.json` at every step — nothing refit, only relabelled.
The result itself and the audit findings live in `status.md`.

wandb was integrated across `train.py`, `calibrate.py`, `run_eval.py`, `metrics.py`
(additive, off by default), and the whole repo migrated to Hydra. Three wandb bugs and
three Hydra behaviors were found and fixed along the way — all recorded in
`docs/STRUCTURE.md`, since they are standing facts about the tooling rather than history.

---

## Stage 1 — planar contact maze

### What was built first (2026-08-12, `41c90d8`)

Environment, physics, guards, a first board, and the full SAC+HER pipeline
(`reward.py`, `gym_env.py`, `train_contact.py`), smoke-tested and gate-verified. Full
derivation and memo citations are in `docs/stage1_env_spec.md`.

**The straight 3-room corridor is the locked plumbing-correctness proof.** Board 90x60cm,
rooms `[0,30]`/`[30,60]`/`[60,90]` in x, portals at x=30 and x=60 both y in [25,35] — a
straight crossing, deliberately no turn. A scripted push-toward-target policy, driven
through the untouched generic `run_episode` + `planner.bfs_route`, completes 0->1->2->goal
in 33/62/20 ticks per leg, zero guard violations, `success=True`. `T_o=200` (>3x the worst
leg) and `episode_budget=600` are derived from that measurement, not assumed — the same
discipline as Stage 0's `h_region`.

**Offset (zigzag) doors: three findings, deliberately not scripted around.**

1. `angular_drag_arm_cm` was a real bug (1.0 -> 6.0). Every push tested before the 3-room
   board was perfectly centered, so rotational-friction damping was never exercised; a 1cm
   offset push spun the object -81 degrees at the old value, -6 degrees at the fixed one.
   Carries forward regardless of the zigzag question.
2. Even fixed, a single continuous single-finger push tolerates only ~1.5cm of lateral
   offset, and this is a genuine physical limit — raising the damping arm to 10/20/60cm
   does not raise the ceiling. Not a remaining tuning gap.
3. **Handoff velocity carryover, observed directly rather than inferred.** At the exact
   tick a non-terminal edge's loose arrival fired, the object was moving ~11.5cm/s and
   rotating 0.13 rad/s — not settled. Feeding that exact exit state into a follow-up edge
   with **no new steering demand at all** still lost contact 31 ticks later. Residual
   motion alone is enough to break a chained push.

**Decision:** stopped chasing this with scripted heuristics (a decelerating, not just
bang-bang, policy still failed, just later). This is the gap SAC+HER exists to close.

**Object-centric observations** are in per memo sec 4.2, verified by direct numerical
translation-invariance (shift the scene 30cm, obs identical to ~1e-6). A hard prerequisite
for one shared push policy to generalize across board locations — implemented before any
training code specifically because of that.

### Runs 1-3: three diagnosed failures (2026-08-12)

100k steps each, both templates ~0% success.

- **Push, run 1: guard-fail-fast was cheaper than a full failed attempt** (~-12 vs ~-300)
  under the original placeholder shaping, so the policy steered the finger *away* from the
  object. This mechanism resurfaced twice more under `w_d` and in the guard sweep before
  `w_m` was raised enough to fix it.
- **Recontact, run 1: gets close (`min_dist` often <2cm) but never settles** — the reward
  has no term rewarding deceleration, only a flat time/effort penalty. Re-confirmed by
  tick-trace three sessions later with much more data. `table_friction` was ruled out as
  the drift cause (its own drag math predicts stopping far faster than observed).
- **Run 2:** push fled the object on tick 1 — the all-zero reward ablation gave SAC nothing
  to reinforce. Recontact got close but flew through at 4-25cm/s.
- **Run 3** (added `w_d`/`w_a`/`w_T`): push moved for real but still 0/40 arrivals,
  `contact_lost` dominant. Recontact logged one lucky mid-training settled arrival that
  did not survive to the final checkpoint — which motivated best-checkpoint saving.

### v13: a handoff doc's claim was wrong; a third bug found (2026-08-12)

**v12 said `RewardWeights` was "confirmed on disk" ablated. On direct read, it wasn't** —
`w_d/w_a/w_F/w_m/w_T` still had non-zero placeholders. Root cause unknown and not
investigated. The actionable lesson became a standing gotcha: **verify a handoff doc's
code claims by reading the file.** Cost one `Read` call; not checking would have cost a
wasted retrain.

Fix then genuinely applied, verified with a `step_reward` smoke test: 20/20 non-arriving
zero-action steps return exactly `0.0`; a synthetic settled `Arrival` returns exactly
`10.0`; a guard violation plus high force still returns `0.0`.

**Third bug: the push curriculum's wall margin was undersized for the finger, not just the
object.** The active finger sits ~6.18cm west of the object center but `_WALL_MARGIN_CM`
was 6.0, sized only for the object's half-diagonal. The 0.18cm shortfall let ~2% of push
resets spawn the finger inside a wall. Traced example: object at `x0=36.10` in room
`[30,60]` -> `active_x=29.92`, 0.08cm past the solid `x=30` wall.

**Fixed generally, not patched:** `_sample_room_xy` takes an `extra_offsets` argument —
points that must *also* clear the margin. With no offsets the math reduces to exactly the
old behavior, so every other call site is unaffected. Verified 0/200 seeds fail the
touch-after-one-tick check, was 1/50.

Also landed: curricula generalized (both fingers/faces/destination room randomized per
episode); `train.py:_region_success` takes a running `max` instead of overwriting `ok`
(a confirmed no-op today, since every caller uses `terminate_on_arrival=True`, fixed
anyway to remove a latent trap). F3 and F2.5's E-a both completed here — see
`docs/TODO.md`.

**Caveat on runs 2-3:** they were launched from code predating the wall-margin fix, so
they include only fixes 1 and 2. Low-frequency bug (~2% of resets), so probably a minor
confound — but flag it rather than treating run 2 as a clean test of all three.

### v15: best-checkpoint saving; the first HER-vs-no-HER split (2026-08-19, `ea76001`)

**Motivation:** run4's push `critic_loss` climbed into the thousands while `actor_loss`
trended steadily more negative. Since SAC's actor loss is `E[alpha*log_pi - Q]`, that is
one phenomenon — the critic's Q-estimate growing — in two curves, not two problems. Given
push's actual weights, the reward this task can produce cannot support Q-values that
large. Read as likely Q-value overestimation, with HER's position-only relabeled reward as
the leading suspect.

**Best-checkpoint saving** (applies to every run since): `ContactPeriodicEvalCallback`
takes `best_model_path` and saves whenever `eval/success_rate` beats its running best.

Two 8-cell ablation sweeps launched (push: `use_her` x `w_d` x seed; recontact: `use_her`
x `w_T` x seed), each cell its own wandb run so the shared-run race cannot apply.

**Outcome: `use_her` split decisively for push** — `false` kept critic/actor loss small
and sane, `true` diverged wildly. Not a shared-cause story between the two settings.

### v16: a real behavioral mechanism, and a sweep bug (2026-08-19)

**Push at `w_m=50`/`100` with `use_her=false` produced the first genuinely real pushing
behavior in the project.** Episodes ran 178-189 ticks (was ~5-6), the object moved 15-21cm
per episode on average (up to 45-49cm; was ~0), and the finger got measurably closer to
goal (best 9.6-19.2cm, was ~23.7cm). **`use_her=true` at identical `w_m` showed zero
change from `w_m=2.0`** — still 100% instant `contact_lost` in ~6 ticks. This is stronger
than "HER adds instability": **HER actively blocks this fix from taking effect.** Treat
`use_her=false` as settled for push.

**Push still doesn't succeed, and now the mechanism is traced, not inferred.** Direction
correlation with the true goal is weak (`w_m=50`: cosine 0.24) to random (`w_m=100`:
0.06), and average progress toward goal was slightly *negative* at `w_m=100`. Phase
analysis found it: the policy sprints early (~6.5cm/s), decelerates to near-zero, and
**frequently parks the object against a wall** (final `y` clustering at ~3 or ~57 on a
60cm board, worse at `w_m=100`). Pinned against a wall the object can't slide, so
`contact_lost` becomes trivial to avoid forever — a cheap way to satisfy the guard
permanently at the cost of ever finishing. **Bigger `w_m` makes this trade more
attractive, not less** — pushing `w_m` higher is the wrong direction.

**Recontact, same sweep:** `use_her=false` essentially unchanged by `w_m` (its guard
rarely fires either way). `use_her=true`: the new overshoot guard fires correctly (9/25
and 3/25 in two cells) but hasn't yet produced more successes. Q0-vs-realized-return gap
stayed large and positive (28-38) regardless of `w_m` — again an HER-relabeling artifact,
not a guard-magnitude problem. **Seed dominates every reward knob tested for recontact
under `use_her=true`, reproduced across two independent sweeps with different axes** —
seed 1 beats seed 0 by a wide consistent margin both times. Arguably more important than
any single reward-knob result: **a reward tweak tested on 1-2 seeds risks just
re-detecting this seed effect.**

**Overshoot guard built:** `RECONTACT_OVERSHOOT_GRACE_STEPS=5` plus a check in
`ContactEnv.step` — not in `recontact_guard`, since guards there are deliberately
target-agnostic and this needs the target. Tracks consecutive ticks inside the loose
radius without settling; past the window, `guard_outcome="overshoot"` flows through the
existing `w_m` path. Smoke-tested against a real trained model, 1/25 episodes caught.

**Settle thresholds investigated, then made sweepable.** `CONTACT_EPS_V_CM_S=0.5` and
`CONTACT_EPS_OMEGA_DEG_S=5.0` turned out to be a judgment call with no derivation, unlike
`k_v`/`angular_drag_arm_cm` nearby in the same spec table. Not baseless: the one measured
bad handoff state (~11.5cm/s, ~7.5deg/s) sits well clear of them — 23x margin on linear,
but only ~1.5x on angular, so there is very little room to loosen the angular one.
Separately, the best recontact policy passes the target at 6.5-9.4cm/s regardless of
`use_her` — 13-19x the threshold, with **no measurable slowdown near the goal versus
anywhere else.** Read as "no braking behavior has been learned at all," not "the bar is
unfair," since even partial braking would show up. Both thresholds became optional Hydra
keys (default `None` -> module constants, every existing caller unaffected).

**Sweep bug, found while presenting results:** `submit_sweep.sh`'s push branch computed
`WM` per array index but never appended `w_m=${WM}` to `EXTRA_OVERRIDE`. **All 8 push
cells trained identically at the default `w_m=2.0`**, not the intended 10/20/30/75 —
confirmed via `meta.txt`'s logged override line and by same-seed `progress.csv` files
differing only in float noise. Recontact's branch was unaffected. Now a standing gotcha:
verify a sweep varied what it claims by grepping the *logged* command, not the launcher.

**Recontact settle-threshold results** (training-time `progress.csv` only, not
tick-traced): `max eval/success_rate` by profile, both seeds — `(0.5,5.0)` control 0.0,
0.0625; `(2.0,5.0)` 0.25, 0.0625; `(4.0,5.0)` 0.25, 0.1875; `(2.0,6.5)` 0.25, 0.0625.
Loosening `eps_v_cm_s` alone coincides with more cells reaching 0.25 than the control did
in either seed; loosening `eps_omega_deg_s` too added nothing beyond that. **Not a finding
yet** — exactly the kind of aggregate pattern that has been wrong before until tick-traced.

### v17: H1 and H3 confirmed by direct measurement (2026-08-19, `a7c153a`)

**Infra:** the `w_m` bug fixed on disk (not yet rerun). Every training sbatch cut from 4
CPUs to 1 — the code is single-process/`n_envs=1`/`DummyVecEnv` and never used more than
one core, which FASRC had flagged as ~194 wasted CPU-hours. Push sweep budget cut 500k ->
250k/cell, since episode length and `eval/ep_rew_mean` both plateau by ~200k.

**Real bug: SB3's `Monitor` wrapper was never applied to any contact run, ever.**
`train_contact.py` hands `SAC()` an already-built `DummyVecEnv`, which skips SB3's
auto-wrap (it only fires on a bare env). So `rollout/ep_rew_mean` and `ep_len_mean` were
silently never logged on any contact run before the one-line fix in `_make_env`.

**H1 (push — "HER grades barely-moving as a free win") confirmed with numbers.** Replaying
real trajectories through HER's own `future` relabeling gives a **99-100% free-win rate**
for a barely-moving policy vs 55% for the one policy that has genuinely moved the object.
New `guard_terminates: bool` flag removes the "bail early, cheap" exit. Combined with a
50x30cm 2-room board and `goal_reward=10` only (100k steps): still 0% success, but a **new,
milder failure mode** — real but small displacement (mean 2.47cm, up to 18.65cm; not the
old ~0cm bail or the wall-parking pattern), free-win rate down to 74.7%, still high enough
to dominate.

**Disambiguating follow-up: is 89% `contact_lost` a policy failure or a physics/guard
limit?** Scripted, no-learning, same env: a **constant** open-loop push direction holds
contact for the first 50 ticks in only 3/30 episodes, and 30/30 eventually exceed the
5-tick grace. A **closed-loop** rule — re-aim at the object's current position every tick,
still zero learning — holds contact **30/30, 100% of every tick, zero violations.**
Settles it: **physics and guard tolerance are fine.** The trained policy isn't doing the
trivial reactive correction the observation already supports (`rel_L`/`rel_R` are already
in the observation).

**H3 (recontact — "HER's `compute_reward` is position-only, so fly-bys score wins")
confirmed with numbers.** A `speed_aware_goal` flag widened `achieved_goal` to
`(x, y, speed)`, requiring the transition's own recorded speed below `eps_v_cm_s`. 100k
steps: still 0% success, but **critic_loss peak dropped 11,028 -> 38 and actor_loss peak
-1,954 -> -324** — H3 was a real, large driver of the Q blowup. Closest approach got
*worse* though (mean 8.85cm vs 0.43-0.91cm for the best position-only cells): "close AND
slow" co-occurring is far rarer for exploration to find than "close" alone, so HER has
fewer true positives at this budget.

**`min_progress_cm` built and calibrated, not guessed.** Zero-action rollouts give
**exactly 0.00cm** displacement every episode — no physics noise floor at all. The
closed-loop scripted push gives the competent distribution: min 2.9cm, median ~10cm. At
5cm, 15% of *competent* episodes (9/60) don't clear it, so real correct pushing would
sometimes look like nothing happened. At 3cm, 59/60 clear it. **Recommended 3cm; set to
1cm per explicit instruction** — more permissive, and since the weak policy's own mean
drift is ~2.5cm, 1cm may still admit much of the free-win problem.

### v18: a systemic HER bug found while auditing the obs space (2026-08-23, `1d72506`)

**Job `40632235` diagnosed by tick-trace.**

- **Push, `restrict_contact_actions=true`, still fails, mechanism now precise.** The clamp
  covers only the outward-normal component, computed once per tick from tick-start state;
  its own "would this open the gap" check reads ~0.00 at the exact tick contact breaks, so
  by its own logic nothing went wrong. Over 40 traced episodes, **62% of first contact
  breaks happen with the finger already >80% of the way to the geometric edge of the
  contacted face (median 94%)** — TANGENTIAL sliding off the corner, which the fix never
  restricts. The rest coincide with a near-saturated raw command (median inf-norm 0.86,
  28% fully saturated) — a fast tangential flick that can separate contact within one
  ~0.04s tick faster than a once-per-tick clamp can track. Left parked, default off.
- **Recontact (1M steps): flat, not slow.** `rollout/success_rate` at noise the entire
  run, `eval/success_rate` 0.0 throughout except isolated 0.0625 blips. `critic_loss`
  stayed in the tens-to-hundreds (vs 11,028 pre-H3-fix), so **this is not a Q-blowup
  story** — `obj_settled` is doing its job. Wall-clock (~5.9hr/1M steps at 47-67 fps,
  single CPU, one gradient step per env step) is exactly what the hardware costs, not
  anomalously slow. Read as an **exploration/curriculum problem**: full task difficulty
  from step 1, sparse reward, one env, no ramp.

**The systemic bug, found by reading SB3 2.9.0's `_get_virtual_samples` directly:**
relabeling only ever overwrites `desired_goal` — `"observation"` is never touched. But
`physics.obs()` bakes the current episode's target into observation's last two slots
(`rel_target`), computed from whatever goal was live at collection time. On a relabeled
transition `"observation"` therefore describes the ORIGINAL goal while `desired_goal` says
something else, for **~80% of every batch** (`her_ratio` at `n_sampled_goal=4`). At real
rollout time the two are always consistent, so the critic trains on a contradiction it
never sees at inference.

Five fixes landed:

1. **Push: `her_buffer.py`** — `PushRelabelSafeHerReplayBuffer` overrides
   `_get_virtual_samples` to recompute `observation`'s target slice from `achieved_goal`
   (which for push already IS the object's absolute position). A ~15-line diff against
   SB3's own method body, easy to audit against the installed source.
2. **Recontact: goal moved to the OBJECT's frame.** This independently fixes a second bug
   — the old world-frame target was sampled once at reset and never updated, so it stopped
   denoting the intended perimeter contact point the moment the object moved. Object-frame
   goals are well-posed under HER (rotation is orthogonal) and dissolve bug 1 for
   recontact entirely, so no custom buffer is needed there.
3. **Recontact: persistent object-disturbance gate.** `obj_settled` only checked
   settledness AT arrival, so a policy could bulldoze the object and let it re-settle just
   before arriving. New sticky `_object_disturbed` flag ANDed into the arrival condition.
4. **Obs hygiene:** dropped `no_contact_steps`/`peak_force` (`OBS_DIM` 21 -> 17) — grep
   confirmed no guard or arrival check reads `obs()`'s output at all, so these were pure
   policy-input noise: unbounded counters next to ~[-1,1]-scale features. Positions now
   divided by board extent, velocities by `v_max_cm_s`, analytically rather than via
   `VecNormalize` (checked SB3's actual behavior: it stores RAW obs and normalizes at
   sample time, so the stale-scaling failure mode doesn't apply — analytic was chosen for
   simplicity and because the goal arrays had to stay raw anyway). `omega` left
   unnormalized: no natural reference scale exists.
5. **Push: inactive finger masked to zero.** It was fully policy-controlled with no
   task-relevant signal, and `push_guard`'s `forbidden_contact` fires on ITS contact — so
   pure exploration noise could tank an episode for reasons the active finger could not
   influence.

**Verification, before trusting:** all 4 gates clean. 7 synthetic checks including
`achieved_xy` provably invariant under a hand-applied 90-degree world rotation (confirms
genuinely object-frame), and `compute_reward` returning exactly `0.0` for a
`settled=True, disturbed=True` transition even with an exact goal match. The load-bearing
check: trained a real tiny buffer, called `_get_virtual_samples` directly, confirmed the
recomputed target slice matched a hand-computed value via `np.allclose` — **and that the
fix is not a no-op**, the relabeled target differing from the stale one by up to 0.38
normalized units on the same batch.

### v19: push's critic-loss blowup found and fixed (2026-08-23, `1d72506`)

**Job `40910275` diagnosed.**

- **Push: identical failure signature to every prior sweep.** `rollout/success_rate` and
  `eval/success_rate` exactly 0.0 in every logged row, both seeds, whole run.
  `rollout/ep_rew_mean` also exactly 0.0 — mathematically forced given the pure-sparse
  ablation and zero real successes, so not itself informative.
- **Recontact: the first physically-verified real success in the project's history.**
  Confirmed by direct rollout (0.000cm/s object velocity, sub-`arrival_eps` distance), not
  the dashboard — the v18 object-frame goal and disturbance-gate fixes work. But success
  **peaks then declines** later in the 1M-step run, correlated with rising `critic_loss`.
  A new, undiagnosed phenomenon; `model_best.zip` already captures the peak.

**Three hypotheses about why push still fails, reasoned through:**

1. **`min_progress_cm=3.0` was too strict.** Confirmed by reading the code: the flag gates
   only HER's *virtual* relabeled reward, since real per-step reward comes from
   `step_reward()`, a separate path that never calls `compute_reward`. With push's reward
   fully sparse and real success essentially never happening, HER's relabeled 80% of every
   batch is the *only* channel carrying learning signal — and 3.0 zeroed that credit for
   any episode moving less than 3cm, i.e. every episode. Lowered to `0.5`, which matches
   `arrival_eps=0.4` and clears the deterministic 0.00cm floor without suppressing 1-3cm
   real progress.
2. **`guard_terminates=false` floods the buffer.** Once `contact_lost` fires (~tick 1-10)
   the episode still runs to the 200-tick horizon with the object sitting still, so HER's
   `future` strategy draws most relabel targets for early transitions from that ~190-tick
   dead tail. Flipped to `true`.
3. **Push's real success requires settled velocity while HER's `compute_reward` was
   already position-only** — a genuine train/eval asymmetry. Tightening the *virtual* side
   would have made an already-too-sparse signal sparser, so this became a separate
   `require_settled: bool` flag instead, loosening the *real* criterion to position-only
   and closing the gap from the other direction. Kept as its own ablation cell.

**Job `40944664` submitted, then its push cells diagnosed early (46k-80k of 150k steps),
all four:**

- `ep_len_mean` climbing ~11-12 -> ~20-25 ticks — a genuine mild positive signal (contact
  held longer before `contact_lost`), the first movement in this metric across any push
  sweep.
- Real success exactly 0.0 in every logged row, all four cells, **including both
  `require_settled=false` cells** — even the loosened criterion never fired.
- **`train/critic_loss` diverging hard in all four** (correlation with timestep 0.57-0.86;
  maxes 18,335 / 1,874 / 3,651 / 2,913). Common to all four, so not caused by the
  `require_settled` ablation. Given `goal_reward=10` and `gamma=0.99`, a legitimate
  bootstrapped Q has a rough ceiling near `10/(1-gamma) ~= 1000` even scoring every step
  forever — the worst cell is 18x past that. **Genuine value divergence, not
  correctly-learned large returns.**

**Two compounding bugs found and fixed together:**

1. **Reward/done mismatch under HER relabeling.** A relabeled transition's `dones` flag is
   copied from its original rollout and never recomputed from the new reward — a known HER
   pathology, not unique to this codebase. A virtual goal that scores the arrival bonus
   still lets the critic bootstrap `r + gamma*Q(next)` past it instead of stopping, which
   got worse once fix 1 above made virtual arrivals common.
2. **`min_progress_cm` was answering the wrong question.** It compared the relabeled goal
   against the *episode's start* — a per-episode gate. But HER relabels per-pair: a
   transition at tick `t` relabeled to tick `t'` is a free win specifically when the object
   was *already* near the goal right before that transition, regardless of the episode's
   total displacement. Conversely a per-episode check cannot tell a genuinely informative
   pair (tick 3 relabeled to tick 150) from a trivial one in the same episode (tick 150
   relabeled to tick 180), since both share the identical comparison. **Verified strictly
   more correct:** a fully-static episode still fails the new per-pair test, so the
   original degenerate case is not reopened.

   Implementation: `compute_reward` refactored into a shared
   `_her_arrived(achieved_goal, desired_goal, info)`. `min_progress_cm` now reads a
   per-tick `info["pre_achieved_goal"]` — captured at the top of `step()` before physics
   advances, exactly what the replay buffer already stores as that transition's own
   `obs["achieved_goal"]` — instead of the removed per-episode `start_achieved_goal`.
   `her_buffer.py` calls `_her_arrived` directly and ORs its result into `dones`, fixing
   bug 1 with the exact same "arrived" definition, so the two cannot disagree.

**Verified:** both motivating examples run through `_her_arrived` directly — a pair whose
pre-transition position already sits next to the relabeled goal scores `arrived=False`; a
pair far from the goal whose trajectory reaches it scores `arrived=True`; the
whole-episode-static case still scores `False`. A full `SAC.learn()` smoke run completed
and every virtual sample with positive reward was confirmed `done=True`. Recontact's path
is untouched byte-for-byte (same settled/disturbed logic, only moved into the shared
helper), confirmed by a recontact smoke run.

**Not fixed for recontact.** It uses the plain SB3 `HerReplayBuffer`, so bug 1's done-patch
was applied only to push's subclass. Its `min_progress_cm` is unset, so bug 2 doesn't bite
it either — but the same reward/done mismatch is structurally present in recontact's HER
path and is a plausible, uninvestigated contributor to the peak-then-decline pattern above.

**Next:** the push rerun in `docs/TODO.md`. Not yet submitted as of this entry.

### 2026-08-23/24: documentation and complexity cleanup

Not an experiment. Compressed docstrings and comments repo-wide toward CLAUDE.md's
<=3-lines-per-function rule (11,239 -> 10,810 lines; prose 2,378 -> 1,955, 21% -> 18%),
removed 22 unused imports, and split this handoff doc into `docs/STRUCTURE.md`,
`docs/TODO.md`, and this file. All four gates re-verified green after every batch.

Three real defects surfaced by the pass, all fixed:

- **`.gitignore` line 28 was a concatenation accident.** The file ended `status.md` with
  no trailing newline, so an appended pattern glued onto it as
  `status.md.claude/settings.local.json` — meaning **neither `status.md` nor
  `.claude/settings.local.json` was actually being ignored.**
- **`contact_templates._off_board` returned True when the object WAS on the board.** Every
  call site negated it, so behavior was correct and only the name lied. Renamed `_on_board`.
- **Two stale comments in `gym_env.py`** still described v19's removed per-episode
  `start_achieved_goal` semantics, and `domains/nav/base.py`'s header still read
  `# systems/base.py` from before the `domains/nav/` move.

### v20: the push rerun had already run; two new mechanisms measured (2026-08-24)

**Correction to v19's closing line and to `docs/TODO.md`'s "immediate" item: the rerun was
submitted and finished on 2026-08-21.** Job `40957220` (4 push cells, 150k each) was
submitted at 15:07, the same minute `40944664`'s push cells were cancelled, and completed
by 16:07. Three days of results sat unanalyzed because all three docs said the opposite.
Its per-run `submit_script.sh` documents both v19 fixes inline, so it genuinely is the
post-fix run — but `meta.txt` records `GIT_COMMIT=a7c153a` for **both** the pre-fix
(`40944664`) and post-fix (`40957220`) runs, because the fixes were uncommitted
working-tree changes at submit time and only landed in `1d72506` two days later. **The
recorded commit cannot distinguish the two runs**; only the copied submit script can.

**The done-flag fix worked, and it is the largest single effect measured on push so far.**
Peak `train/critic_loss`, same config and seeds, pre-fix -> post-fix: 21,719 -> **17.5**
(objhygiene s0), 8,067 -> 549 (s1), 20,056 -> **17.5** (nosettled s0), 8,517 -> 838 (s1).

**But it traded divergence for a dead fixed point, seed-gated.** Both s0 cells decay
monotonically to `critic_loss` 0.67 and sit there from 44k to 150k, `actor_loss` flat at
-1.3, `ent_coef` 0.002 — converged to a stable no-op. Both s1 cells bottom out at 0.85
near 44k then climb back: 0.9 -> 1.3 -> 3.5 -> 16.5 -> **58.6** over the last 50k, still
rising at the end. Also: **v19's "first movement in `ep_len_mean`" was an artifact of the
diverging critic.** Post-fix `ep_len_mean` *declines* (corr with timestep -0.31 to -0.40)
where pre-fix it climbed (+0.39 to +0.66).

**New mechanism 1: `min_progress_cm` erases 98.5-99.5% of push's only learning signal.**
Push's reward is fully sparse (every `w_*` = 0), so HER's relabeled reward is the sole
learning channel. The gate is `|ag_{t+1} - dg| < arrival_eps` AND
`|dg - ag_t| > min_progress_cm`. By the triangle inequality both hold only if the object
moves more than `min_progress_cm - arrival_eps` **in that single tick**. At 0.5 and 0.4
that is 0.1cm; measured median per-tick object displacement over 60 deterministic rollouts
of the trained checkpoints is **0.0098cm**, 10x too small.

Measured over every HER `future` pair in those rollouts, both post-fix seeds:

```
min_progress   signal %    kept vs no gate   survivors' tick speed / median
  (none)       66-67%          100%                  1.4-2.1x
  0.20         20.7-21.6%      31-32%                2.4-4.2x
  0.30          8.6-11.0%      13-16%                3.9-6.1x
  0.40 (=eps)   1.2-3.0%      1.8-4.5%              13-16x
  0.50 (ran)    0.33-1.0%     0.5-1.5%              21-43x
  1.00          0.000%           0%                    n/a
```

Two consequences. **The cliff sits exactly at `arrival_eps`**, and anything at or above
`arrival_eps + max tick displacement` (0.4 + 0.48 = ~0.9cm) gives **provably zero** HER
reward — so v19's pre-fix 3.0 was not merely strict, it was silent, and 3.0 -> 0.5 moved
from exactly 0% to ~1%. **The useful range lies entirely below `arrival_eps`.** And the
~1% that survives is selected for the fastest-moving-object ticks (21-43x median), which
under `require_settled=true` can never be real successes — the fix closed one train/eval
mismatch and opened another.

**Push still has zero verified success.** Direct rollout, 60 deterministic episodes x 4
cells x 2 checkpoints: 0/60 everywhere. `contact_lost` accounts for 57-60 of 60 in every
cell. Median closest approach ~25cm, essentially the starting distance; net object
displacement 0.58-1.65cm. Consistent with v17's scripted result (a zero-learning
closed-loop rule holds contact 30/30), and mechanism 1 explains why the policy never
learns that correction: the channel meant to teach it is 99% empty.

Two smaller push findings. `require_settled=false` is the **only** setting that has ever
made push's success nonzero — `rollout/success_rate` reaches 0.01 in 25 of 3,000 windows
in both seeds, exactly 0 in both `require_settled=true` cells, pre- and post-fix; under
deterministic rollout it is still 0/60, so treat it as unconfirmed. And **`model_best.zip`
is a useless artifact for push**: `eval/success_rate` never beat 0.0, so "best" is just
whichever checkpoint saved first (cells 0 and 2 have byte-identical best-checkpoint
statistics).

**New mechanism 2: recontact's peak-then-decline is systematic, and it is Q-divergence,
not entropy.** All four 1M cells share one shape. Held-out eval success (deterministic,
fixed seeds, 200 evals/cell), mean per 100k bin, averaged over the 4 cells: 0.070, **0.142
(peak)**, 0.063, 0.066, 0.044, 0.029, **0.031 (trough)**, 0.031, 0.050, 0.075. Peak lands
at 120-360k in every cell; the trough is 4.6x down. **Roughly 800k of the 1M-step budget —
4.7 of the 5.9 hours — is spent worse than the 200k checkpoint.**

The training-log correlations pointed the wrong way and the tick-trace corrected them.
`ent_coef` is the strongest anti-correlate of `rollout/success_rate` (-0.63 and -0.72 on
the s0 cells, -0.21/-0.38 on s1, versus -0.33/-0.39 for `critic_loss`), which suggested
SAC's temperature tuner re-injecting stochasticity. **Refuted by direct measurement:**
action std at reset is 0.49-0.64 at the peak checkpoint and 0.47-0.51 at 1M — the final
policy is slightly *less* random, so alpha is not acting through noise. Also note
`ContactPeriodicEvalCallback` runs `deterministic=True` on fixed seeds, so eval-time
exploration noise was never a candidate in the first place.

**What the trace does show,** Q(s0,a0) minus realized discounted return (gamma 0.99, 60
episodes), peak -> final: `gttrue_s0` -6.6 -> **+44.8**, `gtfalse_s0` -2.8 -> **+40.3**,
`gttrue_s1` +53.1 -> +33.2, `gtfalse_s1` -34.8 -> -8.4. On the two s0 cells the critic
goes from roughly calibrated to predicting ~+40 where the policy actually returns ~-1.5.
That is the reward/done mismatch `docs/TODO.md` already named as recontact's leading
suspect — recontact uses the plain SB3 `HerReplayBuffer`, so v19's done-patch was never
applied to it.

**The behavioral failure is over-conservatism, not overshoot.** Every `final` checkpoint
holds object speed at exactly **0.000cm/s** — perfectly undisturbed — while parking the
finger 0.77-1.89cm away, past `arrival_eps` 0.4. The peak checkpoints nudge the object
slightly (0.08-0.53cm/s) and reach 0.49-0.53cm. Read as the v18 sticky
`_object_disturbed` gate plus an inflated critic pushing the policy into a parked local
optimum: don't touch the object, don't arrive either, and be told it is worth +40.
**`gttrue_s0`'s best checkpoint scores 17/60 (28%) under deterministic rollout with the
object at 0.115cm/s** — the project's best verified recontact rate, and it is a real one.

**`guard_terminates` does nothing for recontact; the seed dominates a third time.**
`gttrue_s0`/`gtfalse_s0` trajectories nearly overlap while s0 vs s1 differ substantially,
reproducing v16's finding on a new axis. Standing threat to every n=2 recontact A/B.

**Process finding: 24 CPU-hours of bit-identical duplicate compute.** Job `40944664`'s
recontact cells 4-7 are bit-identical to `40910275`'s cells 2-5 in every learning metric
(verified column-by-column; only `time/fps` and `time/time_elapsed` differ, which is why
the file hashes do not match). Same seeds, same code, same config. Four cells x 5.9 hours
produced no new information. Silver lining: it proves the pipeline is deterministic, so
the peak-then-decline is exactly reproducible.

**New mechanism 3 (found while verifying the temporal-gate fix, and it outranks the gate):
HER's positive examples for push are capped at ~3% of the task distance, for any gate.**
The temporal gate does what it was designed to do — measured on the same pairs as the table
above, `min_progress_ticks=10` keeps 41-76% of arrived pairs at 1.0-1.2x median tick speed
(unbiased), versus `min_progress_cm=0.5`'s 0.33-1.0% at 21-43x. But building it exposed
the deeper constraint. `arrived` requires `|ag_{t+1} - dg| < arrival_eps`, and
`ag_{t+1} ~= ag_t` for a slow object, so **every** pair HER can score positive is one whose
object displacement is at most `arrival_eps + one tick of motion`. Measured against the
prediction:

```
                                 predicted ceiling   measured max over arrived pairs
require_settled=true                   0.85 cm                  0.81 cm
require_settled=false                  0.88 cm                  0.86 cm
median real goal distance at reset:   26.1 cm
=> HER's positive examples cover 3.1-3.3% of the real task distance
```

Median object displacement among arrived pairs is 0.112-0.143cm; among all pairs it is
0.236cm with a max of 5.3-6.3cm — so HER *can* construct pairs spanning real distance, it
just scores every one of them zero. **The critic therefore never sees a single positive
example of moving the object more than ~0.9cm, on a 26cm task.** No threshold on any axis
changes this, because the cap is set by `arrival_eps` and the policy's own object speed,
not by the gate. So the ~99% signal loss reported above is not the gate misfiring: for an
object that barely moves, essentially every reachable relabeled goal genuinely IS a free
win, and the gate was correctly reporting that there is nothing to separate.

**This relocates push's problem from credit assignment to exploration**, and it is
consistent with everything already on record: v17's scripted closed-loop rule holds contact
30/30 with zero learning, the reward is fully sparse with every shaping weight zeroed, and
memo Eq 15's curriculum ramp (`I^(1) ⊂ ... ⊂ I_e`, expanding backward from the target) is
still unimplemented. The object has to move before HER has anything worth relabeling.

**Landed this session, gate-verified 4/4 plus direct checks:**

- `her_buffer.DonePatchedHerReplayBuffer` — the done-flag patch lifted out of push's
  subclass into a shared base with two hooks (`_patch_observations`, `_patch_infos`), so
  **recontact now gets it too**. Verified: 14/14 positive-reward virtual samples come back
  `done=1` on recontact (previously 0/14), 31/31 on push.
- `min_progress_ticks` — temporal gate, wired through `gym_env._her_arrived`,
  `train_contact.py`, and `config/train_contact.yaml`. Push only, since only the custom
  buffer can supply the lag; `_her_arrived` treats a missing lag as ungated, so real
  rollout transitions are untouched. Verified: lag captured over 0-17, injected into
  `info["her_lag_ticks"]` matching exactly, and the gate binds precisely at the threshold
  (lags 0-4 against `ticks=2` -> `[0,0,1,1,1]`, ungated -> all 1).
- `min_progress_cm` kept alongside it for the comparison, with the coupling documented at
  the point of use.
- `slurm/submit_sweep.sh` retargeted (see its header): push 4 cells at
  `min_progress_ticks` {3,10} x 2 seeds as a single-delta A/B against `40957220`'s cells
  0-1; recontact 6 seeds at **300k** instead of 1M, `guard_terminates` dropped as an axis.
  `meta.txt` now also records `GIT_DIRTY` and `GIT_DIFF_SHA` and saves `uncommitted.diff`,
  so the provenance hole above cannot recur.
- Both templates smoke-tested end-to-end through `train_contact.py` itself, not just an
  import check.

**Not submitted.** The sweep is ready; submitting it is the next action in `docs/TODO.md`.

**Sweep rebuilt before submission, after two measurements changed the design.** The version
described above (push tick-gate {3,10} as a single-delta A/B) was replaced. Two findings
during review:

1. **Longer episodes make things worse, not better, and worse specifically with a temporal
   filter.** Same policy at `guard_terminates=false` (200-tick episodes): arrived rate rises
   67.5% -> 80.0%, median object displacement among arrived pairs falls 0.112 -> **0.000cm**,
   and `lag>=10` passes **87.9% of positives whose object path length is exactly 0.000cm**.
   The temporal filter is blind to the dead tail — the exact free-win flood v19 flipped
   `guard_terminates` to `true` to stop. **The two filters fail on complementary cases:** the
   distance filter goes silent when the object moves slowly, the temporal filter admits free
   wins when the object is static for a long stretch. Episode length is also not a free knob
   — the horizon is already 200 and episodes are short because contact breaks at a median of
   tick 6.
2. **`min_progress_cm` compares straight-line displacement, not path length along the
   relabeled subsection** (`goal_dist(dg, pre)`, gym_env.py) — user-identified as a mismatch
   with the intended semantics. Path length is genuinely immune to the `arrival_eps`
   aliasing, since the triangle-inequality bound applies only to displacement. Measured: max
   path length among arrived pairs equals max displacement (0.81cm) in short episodes
   (ratio 1.0x, because over ~12 ticks of slow monotone drift path ~= displacement), and
   1.68 vs 0.81cm (2.1x) in 200-tick ones. So the path-length version is the better-posed
   filter and still useless *now* — `path > 1.0cm` keeps 0.03% of positives — for the same
   root reason the displacement version is: the object travels ~1cm on a 26cm task.

**Consequence for the filter: drop it to `lag >= 1` and nothing more.** With the done-flag
patch in place a free win is a *bounded, correct* terminal anchor — SAC's target is
`r + (1-dones)*gamma*next_q`, so `dones=1` makes it exactly the reward, no compounding. Free
wins are wasted batch capacity, not divergence; the thing that made them poison is fixed.
`lag >= 1` is also not an invention: **OpenAI baselines computes
`future_t = (t_samples + 1 + future_offset)`** — strictly future — while SB3 deliberately
deviates ("our implementation is inclusive: current transition can be sampled"), and v20
measured the cost as 745 lag-0 pairs, 100% scoring tautologically, 10.2% of all positives.
Larger tick thresholds have no named precedent found and are not swept.

**The finding that reframes push, and the actual reason for the new sweep axis.** Measured
over 600 resets of `_sample_push_edge`:

```
same_room_goal_prob   min initial goal distance    p10     median
      0.0 (as run)           12.96 cm            17.82    25.10
      0.5                     0.27 cm             3.96    14.96
      1.0                     0.27 cm             2.90     7.04
```

**At the setting every push run has used, not one episode in 600 starts closer than 12.96cm**
— against HER's 0.81cm positive-example ceiling and a 0.4cm `arrival_eps`. That is **zero
overlap** between what HER can teach and what success requires, which supersedes every
previous explanation for push's failure: no filter, reward weight, or budget touches it. At
`srg=1.0` the p10 is 2.90cm and 0.3% of episodes start already inside `arrival_eps`, so the
**real** reward can fire with no HER involvement at all. `same_room_goal_prob` already exists
and is wired, so this costs nothing to test.

**As submitted-ready:** push becomes a 2x2 on `same_room_goal_prob` {0.0, 1.0} x seed {0,1}
at `min_progress_ticks=1` with no distance filter, `require_settled=false` throughout
(the only setting that has ever produced nonzero push success), baselined against job
`40957220` cells 2-3; `srg=0.0` is the internal control. Recontact is unchanged: 300k x 6
seeds with the done-patch. Preregistered predictions are in the script header — the load-
bearing one is whether the s0 "dead fixed point" survives ~60x more positive samples, which
separates "the critic was starved" from "the actor/exploration is the bottleneck."

Verified before handoff: all 10 cells' overrides resolved through real Hydra and inspected
(`min_progress_cm=None`, `min_progress_ticks=1`, `copy_info_dict=True`, correct buffer class
per template); all three distinct configs smoke-tested end-to-end through
`train_contact.py`; gates 22/22, 27/27, 171/171, 18/18. **Not submitted — held for manual
submission.**

### v21: push sampler fixed; recontact's done-patch is bimodal (2026-08-24)

**Job `41613939`** (push 2x2 on `same_room_goal_prob` x seed at `min_progress_ticks=1`;
recontact 300k x 6 seeds with the done-patch). All 10 cells completed, all recording the
same `GIT_DIFF_SHA=ecc21bc182f13f6a`, so the new provenance fields work.

#### Push: the machinery fixes worked, and then a task-design bug dominated everything

Three of four preregistered predictions held. `critic_loss` settled at **3.8-4.6** instead
of collapsing to the 0.67 dead floor (positives went 1.0% -> 60.6% of pairs), the dead
fixed point is gone (`ep_len_mean` now *rises* 10.2 -> 22.8 where it previously declined,
`ent_coef` 0.010-0.013 not 0.002), `srg=0.0` stayed at 0 success, and `srg=1.0` produced
push's first sustained nonzero reward (`ep_rew_mean` 0.07-0.22). **The starved-critic
diagnosis was correct and the temporal filter plus done-patch fixed it.**

But on the training distribution the policies were **statistically indistinguishable from
doing nothing** — 300 deterministic episodes each: do-nothing 1/300, random 2/300, trained
1/300 and 3/300, Fisher exact **p = 0.62**. `eval/succ_dN` showed the tell: **zero**
success on the ~48 held-out episodes starting within 3cm, where a learning policy should
be strongest. Contact was held 43-47% of ticks against do-nothing's 100%.

**Root cause, and it is task design, not training.** At reset, comparing the push direction
actually available (object - finger, fixed by which face the finger is on) against the
direction needed (goal - object), over 800 resets:

```
                  median misalign   within 45deg   >90deg (pushes AWAY)   straight-push solvable
srg=0.0                 81.5 deg        36.8%             39.2%                    2.1%
srg=1.0                 97.9 deg        17.8%             55.9%                    4.9%
```

The median episode required pushing the object perpendicular to or away from where the
finger could push it. No hand-coded controller beat inaction either: a family blending
"stay in contact" with "push toward goal" scored 1.0% at every blend weight, and as the
goal-directed component grew, **contact retention collapsed 100% -> 42% and displacement
collapsed to zero** — steering needs tangential motion, tangential motion slides the finger
off the face (v18's measured mechanism), contact breaks.

Solving the other ~94% needs walking the finger to another face, which `push_guard`
forbids: `contact_lost` fires after `CONTACT_N_GRACE_STEPS = 5` ticks, allowing
`5 x 0.04 x 20 = 4.0cm` of finger travel, while rounding the corner of a 10x6cm object with
a 1.2cm finger is a ~7.5cm straight line and longer in practice. Off by ~2x. **Repositioning
the finger IS the recontact template, so the curriculum was sampling push->recontact->push
tasks and training them as one edge** — the memo's own decomposition (Fig 3) asserting
itself.

**Also: the face exclusion never did what its docstring claims.** It excludes the face
"nearest the goal" but judges nearest from the *room centre* (same-room branch, to break a
documented circularity: placing the object needs the face, which needs the direction, which
needs the object) or from the coarse `going_east = dst > src` room ordering with y ignored
entirely (cross-room branch). Measured: the finger lands on the goal side in **40.4%** of
cross-room and **56.9%** of same-room resets.

#### The fix: `push_cone_deg`, face and goal drawn consistently

New `_sample_push_edge_coned` plus `_sample_goal_in_push_cone` / `_ray_interval_in_room`.
Whichever of (face, goal) is free is chosen to match the other, per memo Eq 7 (ξ_e contains
the object face, so the face is part of the edge's identity) and Eq 12 (an eastward push's
initiation set puts the finger on the west face):

- **cross-room:** dst is strictly east or west on a left-right board, so the face is
  determined (`west` to push east) and the goal is coned inside dst **and required to be
  reachable by a straight path through the portal** — memo sec 6.4's geometric-reachability
  criterion, which the old sampler never checked.
- **same-room:** the face is free, so all four are sampled and the goal is coned to match.
  This keeps the face diversity Eq 9's shared network needs rather than fixing the face,
  and it deletes the buggy exclusion logic instead of repairing it.
- A floor of `arrival_eps` on the sampled radius means no episode starts already arrived.
- `push_cone_deg=null` keeps the historical sampler **bug included**, deliberately, so it
  remains a faithful control. Verified the null path is unchanged.

**Measured effect (1000 resets/config):**

```
                     median misalign   >90deg   straight-push solvable   median d0   faces seen
BEFORE srg=0.0              85.2 deg    41.1%           2.1%              24.78    E15 N33 S35 W16
BEFORE srg=1.0              98.6 deg    56.3%           4.9%               7.41    E27 N23 S23 W26
AFTER  srg=0.0 cone=30      12.5 deg     0.0%           4.4%              21.83    E49 W51
AFTER  srg=1.0 cone=30      15.4 deg     3.2%          47.6%               2.01    E25 N24 S25 W26
AFTER  srg=0.5 cone=45      18.4 deg     1.1%          20.4%              13.64    E38 N12 S12 W38
```

**The load-bearing validation: the ALREADY-TRAINED policies jump from 1-2% to 34-39% on the
corrected sampler with no retraining.** 200 deterministic episodes per cell:

```
sampler                 median d0   do-nothing   scripted   trained_s0   trained_s1
historical srg=1.0          7.33      0.005        0.020      0.020        0.015
coned srg=1.0 cone=30       2.12      0.000        0.445      0.340        0.390
coned srg=0.5 cone=45      16.14      0.000        0.170      0.120        0.145
coned srg=0.0 cone=30      21.49      0.000        0.040      0.000        0.000
```

**So the policies had learned a real push skill all along; ~95% of training episodes were
unsolvable by any controller and the aggregate metric hid it.** This corrects v20's
"learned nothing measurable" — accurate on the full distribution, wrong conditioned on the
well-posed subset. Note `srg=0.0 cone=30` stays hard (scripted 4%) because at 21.5cm range
even 12.5deg of misalignment misses by 4.7cm against `arrival_eps=0.4` — the long-range
task is **precision**-limited, not direction-limited, so range should be ramped.

Cone half-angle is chosen by measurement, not derived — memo sec 6.4's "a short random or
nominal rollout achieves nonzero success". `arctan(finger_friction) = 36.9deg` is the right
order of magnitude but not the answer: measured achieved deviation from the contact normal
reaches 65-78deg at p90 while contact still holds, because the object rotates during the
push. What actually binds is separation — commanding tangential velocity at or above the
normal component drops contact retention from ~100% to 67%.

#### Recontact: the done-patch is bimodal — best result ever, and worst divergences ever

Six seeds, 300k each. Seed-matched against job `40910275`'s first 300k (stock buffer):

```
seed  eval_max  peak at   eval_final  critic_max    Qgap(final)   det. success (best ckpt)
 s0     0.4375    280k       0.2500          7.7          +2.8         18/60  (30%)
 s2     0.3125    145k       0.1250         47.7          +2.3         10/60  (17%)
 s3     0.3750    145k       0.0000    143,310.2        +283.7         16/60  (27%)
 s4     0.1250    260k       0.0000    776,941.8        +538.9          4/60   (7%)
 s1     0.0625     60k       0.0625    122,549.3             -          -
 s5     0.1875    140k       0.0625      9,946.2             -          -
baseline (stock buffer, seeds 0-1 only): critic_max 171-631, Qgap +44.8/+40.3 at 1M
```

**Two seeds stayed bounded and both are the best recontact has ever been.** s0's
`critic_loss` sits at **0.0** for the whole run, its Q is nearly calibrated (**Qgap +2.8**
against the stock buffer's **+44.8**, a ~16x improvement), and direct rollout gives
**18/60 = 30% success at 0.000cm/s object velocity** — the project's best verified rate.

**Four seeds diverged, and success collapsed exactly when the critic did.** s3 is the clean
demonstration: critic 1 -> 16 -> 810 -> 23,265 across the 150k-300k window while eval
success fell 0.175 -> 0.006, and its final checkpoint scores **0/60** with the finger parked
8.06cm out (its *best* checkpoint, saved pre-divergence, scores 16/60). Bounded critic ->
success rises or holds; diverging critic -> success collapses at whatever step divergence
begins. Note `model_best.zip` earns its keep here, unlike for push.

**On "maxing out early": it isn't, and the appearance is the diverging seeds.** Eval success
per 50k bin: s0 goes 0.006, 0.038, 0.062, 0.119, 0.194, 0.181 — rising monotonically through
250k with its peak at **279,950 of 300,000 steps, 93% of the budget.** s0 is
budget-limited, not converged. s2 plateaus at ~0.18 from 150k. The aggregate "plateau after
150k" is four seeds collapsing, not a learning ceiling. **This partly reverses v20's
budget-cut recommendation:** cutting 1M -> 300k was right for the peak-then-decline
configuration, but with the done-patch the healthy seeds don't decline and s0 wants more
steps. Fix the divergence first, then raise the budget.

**A rigorous bound makes the next fix principled rather than a guess.** With
`w_d=w_a=w_F=w_m=w_T=0`, `step_reward` reduces to `10 * reached_interface`, and arrival
terminates the episode -- so **at most one +10 per episode, hence Q* <= 10 exactly.**
Observed Q reaches 539 (s4 final), 54x the provable maximum, saturating toward
`goal_reward/(1-gamma) = 1000`, the value of collecting the bonus every step forever. So
clipping the TD target at `goal_reward` is exact here, not a heuristic. **Not yet
implemented or tested.**

**Landed after the v21 analysis, gate-verified 4/4, both sweeps built and NOT submitted:**

- `domains/contact/gym_env.py` — `push_cone_deg` plus `_sample_push_edge_coned`,
  `_sample_goal_in_push_cone`, `_ray_interval_in_room`. `null` keeps the historical
  sampler bug-for-bug as a control; verified unchanged.
- `domains/contact/sac_clipped.py` — `TargetClippedSAC`, SB3 2.9.0's `SAC.train()` with one
  `th.clamp(target_q_values, 0.0, target_clip)` added after the entropy term, plus a
  `train/target_clip_frac` diagnostic so "the clamp never fired" is visible rather than
  inferred. Verified: `target_clip=None` gives a **bit-identical parameter digest** to stock
  SAC after 500 steps (13943.811262 both), and the clamp binds (raw targets already hit
  10.000 in a 500-step smoke run).
- `slurm/sweep_push_cone.sh` — 12 cells, `push_cone_deg {null,30,45}` x
  `same_room_goal_prob {0.5,1.0}` x seed {0,1}, 150k. Expanded from the 6 proposed to add a
  second seed, since "seed dominates every reward knob" is this project's most reproduced
  finding and 1 seed/config would risk re-detecting it. ~1h/cell.
- `slurm/sweep_recontact_clip.sh` — 12 cells, `target_clip {null,10}` x seed {0..5}, 1M.
  The `null` arm is simultaneously the "run longer, see where it converges" experiment and
  the control. ~5.5h/cell.
- `slurm/submit_sweep.sh` marked superseded (not deleted; it is job 41613939's record).

Verified for both sweeps: all 24 cells' dispatch simulated, every distinct config resolved
through real Hydra (confirming `push_cone_deg`/`target_clip` actually vary and that every
`w_*` is 0.0 so both templates stay **fully sparse**), and all five distinct configs
smoke-tested end-to-end through `train_contact.py`.
