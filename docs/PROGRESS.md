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

**Committed and pushed at the end of the session.** `main` fast-forwarded from `a7c153a`
to **`a6ad76b`** (2 commits: `1d72506` from the prior session, then this session's work as
one commit) and pushed to `origin/main`; working tree clean. Audited before pushing since
`git add .` is indiscriminate: 35 files, source and docs only — no checkpoints, no `logs/`,
no `wandb/`, nothing credential-shaped. `.claude/settings.json` is now tracked (the
shareable permission allowlist plus a ruff-format hook); `.claude/settings.local.json`
stays ignored.

Two git-hygiene notes worth keeping. `user.name`/`user.email` were unset, so git derived an
author address from the login node's hostname; now set globally to
`Aden McKinney <aden_mckinney@seas.harvard.edu>`. This session's commit was re-authored
before pushing (safe — it was unpushed), but **`1d72506` carries
`amckinney@holylogin08.rc.fas.harvard.edu` and is now published, so leave it** rather than
rewriting shared history.

**Neither sweep is submitted.** That is the next action; see `docs/TODO.md`.

### v22: both v21 sweeps ran — recontact solved, push's cone confirmed (2026-08-25, `a6ad76b`)

**Question.** Does clipping the critic's TD target at `goal_reward` remove recontact's
divergence, and does training on the corrected push sampler beat training on the buggy one?

**What ran.** The two sweeps built in v21, submitted 2026-08-24 14:37 local from a clean
tree (every cell's `meta.txt`: `GIT_COMMIT=a6ad76b`, `GIT_DIRTY=no`).

- Job `41645514` — push, 150k, `push_cone_deg {null,30,45}` x `same_room_goal_prob
  {0.5,1.0}` x seed {0,1}, 12 cells, ~50 min/cell, all COMPLETED.
- Job `41645529` — recontact, 1M, `target_clip {null,10}` x seed {0..5}, 12 cells,
  ~6h/cell, all COMPLETED.
- Job `41643049` (14:23) was a mis-submit, CANCELLED after ~13 min. Ignore its directory.

All numbers below are read from each cell's `progress.csv`; the eval metric is
`eval/success_rate` at `diag_eval_episodes=16`, so one episode = 6.25 pp.

**Result 1 — the clip works, 6/6.** `target_clip=10`: every seed's last-5-eval mean is
0.863–0.975 (best 1.000 on all six), `train/critic_loss` max 0.9–1.6. `target_clip=null`:
one seed (s0) reaches 0.938 with `critic_loss` max 7.7; the other five end at **0.000** with
`critic_loss` max **1.1e9 – 1.2e11**. This replicates v21's bimodality exactly — bounded
critic learns, diverging critic collapses — and shows the clip converts 1/6 into 6/6.
Clipped seeds first cross 0.9 at 56–73% of the 1M budget, then hold, so recontact is now
converged rather than budget-limited.

**Result 2 — the clip is active only at the start.** `train/target_clip_frac` peaks at
~0.97 around 2.8–5.2k steps, is <=0.012 by 20k, and ~0.000 at 1M. So the bound matters only
for the initial bootstrap blowup; afterwards the critic stays inside it unaided. This is
also the preregistered falsifier passing: the stated failure mode was "clip_frac ~0
everywhere yet seeds still diverge," and instead clip_frac was near 1 exactly where it
mattered and only the unclipped arm diverged.

**Result 3 — push: the cone is the whole effect, and `srg=1.0` gates it.** Mean eval over
each cell's last 8 evals, averaged over the two seeds:

```
                srg=0.5   srg=1.0
cone=null        0.000     0.004      <- control, historical face/goal-independent sampler
cone=30          0.059     0.340
cone=45          0.070     0.289
```

Seed spread is tight (cone30/srg1.0: 0.352 and 0.328). `cone` vs `null` is far outside the
eval resolution; `cone=30` vs `cone=45` is not, so do not rank those two. Push plateaus:
mid-run mean (evals 8–16) 0.24–0.34 against 0.33–0.35 at the end, so more steps at this
scale is not the binding constraint. Push cells ran unclipped and did not need it
(`critic_loss` max 15–36, no divergence).

**Result 4, and the honest negative — retraining has not yet been shown to help push.**
v21 measured the *already-trained* (buggy-sampler) policies at 34–39% on the corrected
sampler with zero retraining. Training on the corrected sampler lands at ~0.34. The two
numbers come from different protocols (60-episode deterministic rollout vs the n=16
in-training eval), so they are not comparable as they stand — but nothing here yet shows a
benefit from retraining, only the already-established benefit from fixing the task. A
fixed task and a better policy are separate claims and need the same protocol to compare.

**Next.** (1) Verify recontact under the 60-episode deterministic rollout with the Q-gap
check, and confirm the finger-parking behavior is gone. (2) Evaluate job `41613939`'s and
job `41645514`'s push checkpoints under one identical protocol to settle Result 4. (3) Push's
remaining lever is contact retention/precision, not budget. Details in `docs/TODO.md`.

### v23: both verifications settled; action interface built (2026-08-25)

**Questions.** (1) Does recontact's ~94% survive an independent protocol? (2) Does
retraining push on the corrected sampler beat zero-retraining transfer? (3) If not, is
the action interface the binding constraint?

**What was built.** `eval_contact.py` — the first standalone contact evaluator. Scores a
checkpoint on a **distance-stratified episode set** (N per bin, reset seeds rejection-
sampled once, so every checkpoint sees byte-identical initial states) and reports success
per bin, the `guard_outcome` termination histogram, contact retention, object
displacement, and Q(s0) against realized discounted return. It prints an **env digest**;
two runs are comparable only if the digest matches. `train_contact.py`'s env construction
was extracted to `build_env_kwargs(d)` so both entry points build the identical env
rather than duplicating 20 lines. First test coverage of `ContactEnv`/`Physics`/
`PlanarFingertipWorld` landed alongside (`python test_code.py contact`, 18 checks).

**Result 1 — recontact verified, with a caveat.** 60 episodes, digest `9152a53a6b01`:

```
cell                overall  0-3   3-6   6-9  9-12   12+   Q-gap   Q>10
clip10_s0             0.917 1.000 0.917 0.917 0.833 0.917   +0.10   0/60
clip10_s3             0.783 1.000 0.667 0.750 0.917 0.583   +1.23   0/60
clipnull_s1 (diverged) 0.033 0.000 0.000 0.083 0.083 0.000 +4848.74 54/60
```

s0 holds 83-100% across every distance bin with a critic gap of **+0.10** — essentially
perfectly calibrated, against **+44.8** for the pre-clip runs. The diverged control has
Q mean **4,849** (max 20,950), exceeding the provable bound of 10 in 54 of 60 states, and
times out in 95% of episodes. **The finger-parking behavior is gone**: median final
distance 0.30-0.36cm against `arrival_eps=0.4`, where v20/v21 policies parked 0.77-1.89cm
short. That preregistered check passes, so the `_object_disturbed` gate needs no revisit.
**Caveat:** the prereg bar was >=0.85 in the 12+cm bin; s0 makes it (0.917), s3 does not
(0.583). The n=16 in-training eval had reported s3 at 0.938, so it overstated that seed.

**Result 2 — retraining push does NOT beat transfer. Preregistered prediction fires.**
All four checkpoints on one identical 60-episode set, digest `098fb8afa82e`:

```
checkpoint                       overall  0-3   3-6  6-9  9-12  12+  retention  contact_lost
41613939_2 transferred (s0)        0.133 0.500 0.167 0.000 0.000 0.000   [see below] 76.7%
41613939_3 transferred (s1)        0.083 0.333 0.083 0.000 0.000 0.000   [see below] 83.3%
41645514_6 retrained  (s0)         0.133 0.417 0.167 0.000 0.083 0.000   [see below] 75.0%
41645514_7 retrained  (s1)         0.150 0.583 0.167 0.000 0.000 0.000   [see below] 80.0%
```

**CORRECTION (v25).** The retention column originally printed here (0.203-0.262) was
wrong: `eval_contact.py` read the active finger ONCE before the episode loop, but `reset`
re-samples it per episode, so retention was counted against the wrong finger on most
episodes. Fixed in v25; re-measured, raw push retention is **0.52-0.61**. Success,
termination and Q figures never used that variable and are unaffected, so the headline
(retrained 0.142 vs transferred 0.108, contact_lost 75-83%) stands.

Retrained mean 0.142 vs transferred 0.108 — ~2 episodes in 60, inside the within-arm seed
spread (0.017 and 0.050). **The cone fix bought a measurable task, not a better policy.**
Also settled: `contact_lost` ends 75-83% of episodes in every arm, no episode reaches the
200-tick horizon, and median displacement is ~2cm regardless of goal distance. One push
option is worth ~2cm of object motion, which is exactly why it succeeds only under 3cm.

**Result 3 — the action interface, built and falsified-against.** `action_interface`
(`finger_velocity` default, bit-identical; `contact_frame` new, push-only, validated to
raise rather than be ignored). Under `contact_frame` the active finger's two components
are (push along the inward face normal, slide along its tangent), re-derived **every
physics substep** under two soft constraints: never open the gap faster than the object
recedes, and cap tangential speed at `slip_limit`. The motivation is structural —
`PlanarFingertipWorld.step` held one command across all 20 substeps, so a 25Hz command
could slide the finger 0.8cm across a face before anything reacted. Contact can still be
lost by walking off a corner, so the `contact_lost` guard stays meaningful (chosen
deliberately over pinning the contact point, which would make the guard vacuous).

Scripted falsifier, 60 episodes, **no training**:

```
controller                        retention  displ_cm  contact_lost
finger_velocity  straight            1.000      7.98       0.0%
contact_frame    straight            1.000      7.98       0.0%
finger_velocity  + steering          0.838      3.55      70.0%
contact_frame    + steering          0.996      9.27       0.0%
```

The straight-push rows being **identical** is the right null result: the constraints only
bind once the controller steers, confirming the plumbing is inert when it should be. Under
steering — what the policy actually has to do — the raw interface loses contact in 70% of
episodes and moves the object 3.55cm; `contact_frame` holds contact and moves it 9.27cm.
`slip_limit=0.5` was picked by measurement over {0.25, 0.5, 1.0} (displacement 8.25 /
9.27 / 7.98), per memo sec 6.4, not from `arctan(finger_friction)=36.9deg`.

**Gates:** static 22/22, geometry 27/27, **contact 18/18 (new)**, test_option_graph
171/171, fixture_eval 18/18. All three sweep arms smoke-tested end-to-end through real
Hydra; all 12 cells' dispatch simulated and verified to vary what they claim (the v16
lesson).

**Next.** `slurm/submit_sweep.sh` was retargeted to this experiment and is NOT submitted.
16 cells: a 4-arm ablation — raw / clamped (the same rule at 25Hz) / contact_frame at
slip 0.5 and 1.0 — x srg {0.5,1.0} x seed {0,1}. The `clamped` arm is what lets the sweep
attribute an effect between the clamp RATE and the reparameterization; the two slip levels
test whether a learned policy wants more tangential authority than the scripted optimum.
Score it with `eval_contact.py`, never the in-training eval. The file previously held job
41613939's launcher; that record is preserved in `logs/sweep_41613939/*/submit_script.sh`,
10 byte-identical copies verified before the overwrite.

### v24: local visualization + storage tooling (2026-08-26)

**Question.** Keep wandb and local disk from filling up, and get local-only
visualizations including videos of policy evals.

**What the measurement changed about the request.** Remote wandb is **155 runs and
0.25 GB** against a 100 GB tier — not a storage problem at all, only dashboard clutter.
Local `logs/` is 1.2 GB of which **1.1 GB is 290 SB3 checkpoints** at 3.26 MB each; local
`wandb/` is 351 MB. So the thing to manage is checkpoints, and home has 72 GB free, so
this is growth hygiene (~104 MB per 16-cell sweep) rather than an emergency.

**Retention is not a pure storage question.** v23's transfer-vs-retrained result was only
possible because job 41613939's checkpoints from two days earlier still existed. A
date-cutoff purge would have destroyed that experiment before it was run. So the pruner
deletes only what the data itself proves is redundant.

**Built.**
- `eval_contact.py` gains `eval_video` / `eval_summary_png` / `eval_video_n` /
  `eval_video_fps` / `eval_media_dir`, all defaulting off. Videos re-run the *scored*
  episode collecting Snapshots, so a rendered episode IS the episode in the table.
  Episode selection prefers arrivals + `contact_lost` failures + the worst final-distance
  case rather than the first N, since the failures carry the information. Because the
  stratified seeds are fixed by the env digest, **episode k is the same initial state
  across every checkpoint**, so arm-vs-arm videos are directly comparable. ~8 KB per mp4.
- `tools/compare_sweep.py` — one figure and table across a sweep's cells, and it
  **refuses to plot when env digests disagree**, which is the check that would have caught
  the v22 confound immediately.
- `tools/prune_runs.py` — dry-run by default. Deletes `model_best.zip` only where that
  cell's `progress.csv` shows `max(eval/success_rate) == 0.0` (per the callback gotcha
  those files are just the first eval snapshot) and only when `model.zip` also exists, plus
  an explicit superseded-job list. Never touches `progress.csv`/`meta.txt`/
  `submit_script.sh`/`run.*`, never touches jobs still in the queue, honours a `KEEP_JOBS`
  list, and writes a deletion manifest. **Dry run: 123 files, 0.38 GB.** Not applied.
- `tools/prune_wandb.py` — dry-run by default; matched exactly **1** junk remote run, which
  confirms the remote side needs nothing. Reports local `wandb/` and points at
  `wandb sync --clean` rather than `rm -rf`.

**Two things caught by measuring rather than assuming.**
1. `visualize.py` had **zero callers** — the same setup that let `plots.py` ship a missing
   `import pandas`. Exercised on real rollout data before wiring it in; all three functions
   work. Now gate-covered (`test_code.py contact`, 22 checks) so it cannot rot again.
2. The first version of `prune_wandb.py` reported local `wandb/` at 1127 MB against `du`'s
   351 MB. Neither was wrong: the `.wandb` files are **sparse**, so apparent size is 3x
   disk usage. A storage tool must report `st_blocks * 512`, not `st_size`, or it
   overstates by 3x. Fixed; now reports 331 MB against `du`'s 351 MB (the gap is the
   top-level log files outside run dirs).

**Gates:** static 22/22, geometry 27/27, contact **22/22** (4 new renderer checks),
test_option_graph 171/171, fixture_eval 18/18.

**Not done, deliberately:** nothing was deleted — both pruners are dry-run and deletion is
the user's call. No periodic checkpointing (it would multiply the 1.1 GB this exists to
contain). No automatic pruning inside sweep scripts, which is how provenance disappears
silently. wandb logging stays on, no-media rule unchanged.

### v25: the action interface is push's constraint — contact_frame works (2026-08-26)

**Question.** Job 42007967, the 16-cell interface ablation. Does `contact_frame` beat the
raw interface, and is it the reparameterization or the clamp rate that matters?

**Two bugs found before the result was trustworthy. Both were mine, both in v24 tooling.**

1. **The batch scoring passed only the TASK overrides**, so every cell — including the
   eight `contact_frame` cells — was scored with the default
   `action_interface=finger_velocity`. Those policies emit (push, slide); the evaluator
   interpreted them as (vx, vy). The first table produced looked like a decisive negative
   for `contact_frame` (success halved, `contact_lost` up to 97%) and was entirely an
   artifact. **Exactly the v16 failure mode: a variable that never reached the command.**
   Caught by replaying one checkpoint on the same seeds and getting a different
   termination histogram than the stored json.
2. **The active finger was read once per checkpoint, not per episode.** `reset` re-samples
   it, so contact retention was counted against the wrong finger on most episodes. This
   bug was present from eval_contact.py's first version, so **v23's retention column is
   corrected above** (0.203-0.262 -> 0.52-0.61 for raw push).

Fix for (1) has two parts: the scoring script now reads each cell's own interface out of
its `meta.txt`, and the **env digest now covers the task only** — reset sampler, board,
reward, horizon — with `action_interface`/`slip_limit`/`restrict_contact_actions` excluded
and recorded separately. Those change how the policy's numbers are interpreted, not what
the task is, so two arms of an interface ablation must stay comparable.

**Result, 60-episode common set, digest `1f402df37eef` (means over 2 seeds):**

```
arm            overall   0-3    3-6    6-9   9-12    12+   retention  contact_lost
raw              0.121  0.458  0.125  0.000  0.021  0.000    0.52-0.61   75-83%
clamped          0.117  0.438  0.125  0.000  0.021  0.000    0.67-0.71   75-83%
cf_slip0.5       0.229  0.625  0.167  0.167  0.104  0.083    0.94-0.96    5-17%
cf_slip1.0       0.292  0.625  0.312  0.188  0.084  0.250    0.90-0.96    0-13%
```

- **The preregistered headline claim holds.** The 6-9cm bin goes from 0.000 in all eight
  raw/clamped cells to nonzero in seven of eight `contact_frame` cells, and the **12+cm bin
  is nonzero for the first time in the project's history** (0.250-0.417 in four cells).
- **`contact_lost` collapses from 75-83% to 0-17%**, and retention rises from ~0.55 to
  ~0.95, essentially reaching the scripted controller's behavior.
- **The attribution arm did its job.** `clamped` improves retention (0.67-0.71 vs
  0.52-0.61) but its success is level with `raw` (0.117 vs 0.121). So **the
  reparameterization is what matters, not the 25Hz-vs-500Hz clamp rate** — holding contact
  is necessary but not sufficient; the policy also has to be able to *steer* while holding,
  which only the contact-frame action gives it.
- **The learned policy wants more tangential authority than the scripted controller did.**
  `slip 1.0` (0.292) over `slip 0.5` (0.229), where the scripted sweep had preferred 0.5.
  Including both levels was the right call. **But seed spread is wide and overlapping**
  (slip1.0 0.183-0.417, slip0.5 0.150-0.333, n=2), so this is suggestive, not established.

**Gates:** static 22/22, geometry 27/27, contact 22/22, test_option_graph 171/171,
fixture_eval 18/18.

**Next.** More seeds before ranking slip levels; push is no longer flat-lined beyond 3cm,
so the range curriculum (memo Eq 15) becomes worth running; and the edge-definition
fallback is no longer the leading option, since a push edge can now move the object
further than ~2cm.

---

## v26 — 2026-08-26 — Coulomb slip, the free finger, and E3 killing the settle sweep

**Question.** Three things, in order of what they cost. (a) Is `slip_limit` defensible as a
tuned constant, or should the tangential budget come from friction? (b) The masked inactive
finger is servo-held in the object's path — is that really what `forbidden_contact` is
measuring? (c) Do push failures overshoot the goal or never reach it?

**(a) slip is now derived, not tuned.** `slip_limit` was a free tangential ceiling
(a fraction of `v_max`) picked by sweeping. Replaced with `slip_model=friction_cone`
(now the default): the tangential budget is `mu * push * v_max` with `mu = finger_friction`
— the same coefficient pymunk already uses for that contact, so there is one number rather
than two that can disagree. It scales with the normal push and vanishes when the finger
stops pressing, which a flat ceiling does not. The worst-case command sits exactly on
`arctan(mu) = 36.87deg` off the face normal (asserted in the gate). `slip_model=speed_fraction`
keeps the superseded formula bit-exact, because the archived `sweep_42007967` checkpoints
have to stay replayable under the interface they were trained on.

Note this **supersedes v25's reasoning** for picking slip by measurement. The observation
that stood behind it — achieved deviation reaching 65-78deg at p90 while contact held — is
still true; it just means pymunk's solver is more permissive than a Coulomb contact, not
that we should command motions a real finger could not make.

**(b) the free finger: mechanism confirmed, zero training.** `mask_inactive_finger=true`
does not remove the inactive finger, it leaves it **servo-held wherever it spawned**.
Harmless at 2cm pushes; not at 6-10cm. Replaying the v25 SOTA checkpoint (cell 15, frozen
policy) under a 60deg spawn cone centred behind the object's travel:

```
disengaged_away_deg   success   forbidden_contact   horizon   contact_lost
null (uniform ring)     0.433      8/60 (13.3%)    22 (36.7%)   4 (6.7%)
60deg (behind travel)   0.333      1/60 ( 1.7%)    36 (60.0%)   3 (5.0%)
```

**`forbidden_contact` 13.3% -> 1.7% with the policy frozen** settles the diagnosis: it was
the object arriving at a stationary fingertip, not a bad grasp at reset. Success *falling*
is **not** evidence against the change — the policy was trained on the uniform ring and is
off-distribution here, and 26-vs-20 arrivals in 60 is ~1.6 standard errors. The two runs
also have different digests by construction (the reset sampler is a task key), so they are
matched on `d0` bins but are not the same episodes. Whether placement *helps* needs training.

**Unmasking cannot be tested this way** and is not claimed to be: replaying a masked
checkpoint unmasked would animate two outputs it never learned to control. It goes in the
sweep. Per decision this round, the `forbidden_contact` guard stays terminal-only with
**no `w_m` term**, so the `unmask` arm measures the guard alone.

**(c) E3 killed the settle sweep before it ran.** `eval_contact.py` now records closest
approach and its tick per episode. On the SOTA checkpoint's 34 non-arrivals:

```
within arrival_eps (0.4cm):  reached  0 (0%)
within 1cm:                  reached  8 (24%) -> overshot 4, held-but-unsettled 4
min_dist   p10/p50/p90       0.60 / 3.75 / 10.97 cm
final_dist p10/p50/p90       1.11 / 5.36 / 12.88 cm
mean (final - min) 1.40 cm   median closest-approach tick 50 (of 200)
```

**No failing episode ever came within `arrival_eps`.** Failures never get close, and give up
only 1.40cm between closest approach and where they stop. That is an **aiming** problem, and
neither `require_settled` nor a longer horizon can fix it — they get nearest at tick 50 of
200 and then wander. The planned 12-cell `horizon x require_settled` sweep (E4) **is not
being run**. Running E3 first, at zero training cost, is what caught this.

**Also.** `mask_inactive_finger` is classified as an **interface** key in `eval_contact.py`
(it decides whether two of four outputs do anything), while `disengaged_away_deg` is a
**task** key and stays inside the digest.

**Built.** `slip_model` + `_tangential_speed`, `mask_inactive_finger`, `disengaged_away_deg`
(null path bit-identical, asserted), E3's `min_dist`/`min_tick` + `overshoot_report`, and a
rewritten 15-cell sweep (arm {base, legacy, unmask, place, unmask_place} x seed {0,1,2}).
All five arms smoke-trained.

**Gates:** static 22/22, geometry 27/27, contact 35/35, test_option_graph 171/171,
fixture_eval 18/18. (`ruff` is not installed in the `tsmc` env — lint gate unrunnable.)

**Next.** Submit the 15-cell sweep. If no arm beats `base` outside seed spread, the free
finger was not push's binding constraint and the next target is the aiming problem E3
exposed, not a fifth fix to the contact interface.

## v27 — 2026-08-26 — the free finger scored: placement is cosmetic, unmasking is harmful

**Question.** Job `42056320`, 15 cells, `{base, legacy, unmask, place, unmask_place} x seed
{0,1,2}`, contact_frame, 150k. Is the servo-held free finger a real constraint on push, and
what did adopting the Coulomb slip law cost? Preregistered in the launcher header;
`forbidden_contact` was named the primary signal because success gaps under ~0.15 are
unreadable at 3 seeds.

**What ran.** All 15 cells COMPLETED at `8165284`, `GIT_DIRTY=no`. Scored with a new
`tools/score_sweep.py`: interface keys from each cell's `meta.txt`, task keys pinned, two
digest groups (A `4c221f0405c6` uniform ring, B `7b7d59c82155` 60deg cone), plus `base`
transferred into group B. **Both `model_best.zip` and `model.zip` were scored** — 36 evals,
34s each, on the `shared` partition. Paired bootstrap over the 60 shared stratified
episodes, 20k resamples.

**Results.**

| arm | `model_best` | `model` | fc% (best/final) | horizon% | reten | Q gap | 12+cm |
|---|---|---|---|---|---|---|---|
| `base` | 0.150 | 0.172 | 15.0 / 12.2 | 61-66 | 0.98 | +5.06/+4.35 | 0.000 |
| `legacy` | **0.294** | 0.233 | 23.9 / 22.2 | 42 | 0.91-0.94 | **+2.69/+3.14** | **0.083** |
| `unmask` | 0.178 | 0.200 | **31.7 / 26.7** | 47-50 | 0.99 | +3.95/+3.60 | 0.028 |
| `place` | 0.156 | 0.228 | **0.0 / 0.0** | 71-72 | 0.97 | +4.52/+3.60 | 0.000 |
| `unmask_place` | 0.167 | 0.239 | 22.8 / 20.6 | 52-58 | 0.98 | +4.44/+3.83 | 0.000 |
| `base` -> group B | 0.150 | 0.211 | **0.0 / 0.0** | | | | |

1. **`place` kills `forbidden_contact` outright: 0/180 episode-scorings, both checkpoints.**
   The prediction was ~2%. The `base`-transferred control is also 0/180, which attributes it
   to the spawn and not the policy.
2. **`unmask` fails its own preregistered threshold.** `forbidden_contact` went UP to
   26.7-31.7% against the 19% bar, so the masking comment's warning still holds at 150k.
   `unmask_place` is not the predicted harmless redundancy: unmasking *reintroduces* the
   collision placement removed (0.0% -> 20.6%). **Keep masking; close out `unmask`.**
3. **The transfer eval killed the placement success claim.** Naively `place` 0.228 vs `base`
   0.172 reads +0.056. Against the correct control, `base` under group B's own reset at
   0.211, it is **+0.017 [-0.039,+0.078]** — nothing. Placement made an easier task. This is
   exactly the read the transfer eval was added to provide.
4. **Only `legacy` beats `base` on success, and only on one checkpoint:** +0.144
   [+0.078,+0.217] on `model_best`, +0.061 [-0.000,+0.122] on `model`.
5. **Pooled against v25's rescored cells the slip-law cost is solid.** 7 `speed_fraction`
   cells (v26 `legacy` x3 + v25 `cf_slip1.0` x4) vs 3 `friction_cone` cells, identical
   digest and identical 60 episodes: **+0.126 [+0.061,+0.194]** (`model_best`) and **+0.080
   [+0.022,+0.137]** (`model`). Both exclude 0. `base` sits below BOTH superseded ceilings
   (v25 `cf_slip0.5` 0.200-0.221, `cf_slip1.0` 0.263-0.267). Confound to state: the v25
   cells trained at `a6ad76b` and varied `same_room_goal_prob` in training, so this is
   corroboration, not one clean experiment.
6. **`legacy` is the best aimer**, which is the mechanism: 39% of its failures reach within
   1cm against `base`'s 19%, median closest approach 1.42cm against 3.54cm. Consistent with
   more tangential authority, and with its lower retention (0.909).
7. **E3 replicates in all five arms, both checkpoints, ~1400 failing episodes: 0% ever
   reach `arrival_eps`.** `horizon` is now the dominant failure at 42-72% with median
   episode length 200 in every bin >=3cm. Aiming is confirmed; the free finger was not
   hiding it.

**Failure modes found.**

- **`model_best` and `model` disagree on the arm ordering** (`legacy` first vs
  `unmask_place`/`place`/`legacy` level). Scoring only one would have given a confident
  wrong answer.
- **Nothing converged at 150k.** The 100-125k eval bucket beats the 75-100k bucket in
  **15 of 15 cells** (sign test p ~ 3e-5). `ep_len_mean` went from v25's 11-14 ticks to
  78-129, so the same 150k buys ~8x fewer episodes. This is the likely cause of the
  checkpoint disagreement. Note v25's "plateau by ~50k" referred to the 200-tick episode
  horizon, a different budget from the training clock.
- **The in-training n=16 eval inflates success ~2x**: `base` 0.323 -> 0.150, `legacy`
  0.431 -> 0.294, `place` 0.413 -> 0.156.
- **The critic gap tripled.** `base` +5.06 against v26's recorded +1.77 (Q mean 5.93 vs
  realized 0.81). No bound violation (max 9.63 < 10, 0/60 over). Inversely correlated with
  success across arms — `legacy` has both the smallest gap and the best success.
  Correlation only; v20's lesson applies.
- **`base`'s seed sd was exactly 0.000 on `model_best`** (9/60 three times), which is the
  v20 duplicate-cell signature. Checked: the three success *memberships* differ (4-6
  episodes) and mean |dQ| is 0.9-1.4, so the policies are distinct and the equal counts are
  coincidence. All three succeed on the same low-index episodes, which is itself the aiming
  story — solvability is set by the task, not the seed.

**A protocol break, found by replaying a stored result.** The v25 SOTA checkpoint replayed
at 0.4333 against its stored 0.4167, with the digest moved `1f402df37eef` ->
`4c221f0405c6`. Cause: **v25's scoring pinned `same_room_goal_prob=0.5`; the v26 launcher's
documented command says `1.0`.** Forcing `srg=0.5` at HEAD reproduces v25's stratified set
**60/60 bit-identical**, so status.md's "`disengaged_away_deg=null` is bit-identical" claim
is CORRECT and the suspected extra RNG draw does not exist. But `srg=1.0` is a materially
easier distribution — mean initial goal distance 7.59cm vs 9.15cm, and the 12+cm bin's max
drops 29.3cm -> 18.9cm. **Decision: pin `srg=1.0` going forward** (it is what every v26
cell trained at) and restate v25 on it; job `42076377` did, giving the row-5 numbers above.
v25's published 0.229 / 0.292 / best 0.417 are on the old distribution and must not be
quoted beside these.

**Next.** Adopt `place`, keep masking, drop `unmask`. Then aiming, with the budget folded
in: `base` and `legacy` both carried as arms at 400k so the slip-law cost and the
convergence question are answered in the same job. **Correction to the recorded plan:
`push_cone_deg` is a half-ANGLE, and `_sample_goal_in_push_cone` samples the goal radius
`uniform(lo, hi)` out to the wall-padded room boundary with no cap.** status.md's "the cone
sampler already takes a radius, so it is a schedule on one number" is wrong — memo Eq 15's
range curriculum needs a new parameter (clamp `hi`), not just a schedule.

## v28 — 2026-08-27 — the training distribution was the mismatch; the cone comes off

**Question.** Four changes were asked for directly: turn the critic clip on for push,
remove the extra friction cone and let the solver own friction, ablate a 3cm goal-distance
floor and an adjacent-room-only goal spawn, and extend the budget to 400k. Before building
them, re-derive v27's numbers from the raw evals and check what they support.

**v27 reproduces exactly.** All 36 eval JSONs re-scored from
`logs/eval/v26_42056320/`: the arm table, the per-bin columns, `forbidden_contact` 0/180
for `place` and for the `base` transfer, and the paired bootstrap
(`legacy - base` +0.144 [+0.078,+0.211] `model_best`, +0.061 [+0.000,+0.122] `model`) all
land on the recorded values. Both digest groups also turn out to share **identical `d0` to
1e-6 on all 60 episodes**, so the group A/B transfer comparison is matched on the goal and
differs only in the free finger's spawn — stronger than "matched on `d0` bins".

**THE FINDING: push trains on a much easier distribution than it is scored on.** Measured
at HEAD, 400 resets per setting, against the fixed 60-episode benchmark:

```
training, srg=1.0 (every push sweep to date)   min 0.4  median 2.1  mean 3.0  64% <3cm   4% >9cm
the benchmark it is graded on                  min 0.4  median 7.2  mean 7.6  20% <3cm  40% >9cm
```

The policy has barely seen a long push. `stratified_seeds` deliberately searches for goals
to fill five distance bins, so the benchmark is *supposed* to be harder — but nobody had
measured how much, and 64%-vs-20% under 3cm is a bigger gap than any effect measured in
v25-v27. This is a more direct account of "long pushes never work" than the slip law, the
free finger, or aiming.

**Three checks on v27's reasoning.**

1. **E3's headline is true by construction.** "0% of failing episodes ever came within
   `arrival_eps`" cannot be otherwise: `require_settled=false` makes arrival
   `d < arrival_eps`, and arrival terminates. Confirmed in the data — over 2,160 episode
   scorings the minimum closest approach among failures is **0.4010cm** and the maximum
   among successes is **0.4000cm**, a hard cut at the threshold. The `arrival_eps` row of
   `overshoot_report` is dead by construction under the pinned protocol. The *conclusion*
   survives on the live numbers (median closest approach 3.09cm, 18% of failures within
   1cm), but the line quoted as proof is not proof, and it was used to cancel a 12-cell
   sweep. Rewrite the row or delete it.
2. **The miss is about half shortfall, not mostly aiming.** Decomposing each failure's
   start/end/goal triangle (free, from data already on disk), failures with `d0>=3cm`,
   median `d0` 9.5cm: `base` travels **3.70cm (48%)** toward the goal with **2.26cm**
   sideways error; `legacy` **6.09cm (73%)** with **2.13cm**. So `legacy`'s advantage is
   almost entirely *travelling further*, not aiming better — which revises v27's "`legacy`
   is the best aimer".
3. **An untrained network scores 0.254 on the in-training eval.** Every cell's first two
   evals precede the first gradient step (`learning_starts=10000`); across 15 cells they
   average 0.254 against a final 0.411. No trivial-baseline number exists for the
   60-episode benchmark. Also: the "still rising in 15/15 cells" claim is 14/15 strictly
   (one tie), and the **125-150k bucket beats 100-125k in only 9/15** — mean gain drops
   +0.078 then +0.021, so the curve is flattening, and 400k will buy less than "still
   rising" suggests.

**Also measured: the 0-3cm bin carries a fifth of the score and is nearly trivial.**
Pooled over all 36 scorings, success is 0.856 under 1cm and 0.009 beyond 12cm; median
object displacement in a *successful* episode is 0.95cm. Restricting to goals >=3cm moves
`base` 0.150/0.172 -> **0.042** and `legacy` 0.294/0.233 -> **0.174/0.153**, i.e. the gap
goes from 2x to 4x. Report both from now on; the restriction is a free reweighting of
episodes already scored, no re-run and no digest change.

**Built.**

- **The enforced friction cone comes off the default.** `slip_model=speed_fraction`,
  `slip_limit=1.0` is now the default, which leaves no tangential limit but the clamp of
  the whole command to `v_max` and "no pulling". Rationale: the finger is a force servo and
  its contact with the object is a native pymunk contact at mu=0.75, so **the solver
  already resolves slip**. `friction_cone` caps the *command* at `mu*push*v_max` on top of
  that — a second friction model over the solver's own. It forbids deliberate slip (a real
  finger slips constantly; v25 measured 65-78deg deviation while contact held) and it
  **freezes the finger entirely at push=0**, so the finger cannot slide along a face
  without also driving the object. `friction_cone` is kept as an ablation arm, not deleted.
  Supporting measurement, success on goals >=3cm, one eval set (digest `4c221f0405c6`),
  14 cells: **cone 0.042 < flat 0.5 0.109 < flat 1.0 0.170**, monotone, and every cone cell
  below every flat cell. Confound: 8 of the 11 flat cells are v25's, trained at `a6ad76b`
  with `same_room_goal_prob` varying in training. The clean within-sweep comparison is
  3 vs 3 with perfect separation, exact one-sided p = 0.05 — the smallest 3-vs-3 can give.
- **`push_range_min_cm`** (push only, cone sampler only): a floor on the goal's distance
  from the object at reset. Rays too short to hold the floor are **rejected, not clamped**,
  which would pile goals onto `hi`. The first version leaked — 3.7% of goals were still
  under a 3cm floor, via `_sample_push_edge_coned`'s `_sample_room_xy` fallback when the
  cone cannot reach; the fallback now redraws until it clears the floor. `None` is
  bit-identical (asserted): with no floor the fallback loop draws once and breaks.
- **Gate: 35 -> 41.** Three checks that the default slip law is `speed_fraction`/1.0 and
  that it slides at `v_max` where the cone freezes, at every push level; three that the
  floor holds over 200 resets, that `None` is bit-identical, and — following the repo's own
  rule about tests that never fire — that **the unfloored sampler really does draw
  sub-floor goals** (131/200 under 3cm), so the floor check cannot pass against a broken
  implementation.
- **v28 sweep launcher**, 18 cells = `{full, noclip, nomin, cone, cross0, cross50}` x seed
  `{0,1,2}` at 400k. Each arm is `full` minus exactly one factor. `disengaged_away_deg=60`
  is **adopted in every cell**, not tested, so there is one digest group and no transfer
  control is needed this round; the reference to beat is `base` scored under that same
  reset in v27, 0.150/0.211 overall and 0.042/0.069 on goals >=3cm.

**Preregistered predictions (BOTH WRONG -- see the results section below).** (1) `full`
beats `nomin` on goals >=3cm and loses to it under 3cm. (2) `cross0` sees near-zero
successes and produces no learning signal, because success beyond 12cm is 0.009 today and
the reward is fully sparse.

**Verification before submission.** All 18 cells expanded from the launcher itself (not a
retyped copy — a retyped copy had a typo the file did not, which is v16's failure mode in
miniature); each differs from `full` in exactly one key. All six arms smoke-trained to
completion. `target_clip` read back out of each saved checkpoint: `noclip` `None`, the
other five `10.0`. Sampler settings confirmed behaviourally, not by reading the command:
`full` min 3.0 median 4.8, `nomin` min 0.4 median 2.1, `cross0` min 12.3 median 21.6,
`cross50` min 3.0 median 13.9.

**Gates:** static 22/22, geometry 27/27, **contact 41/41**, test_option_graph 171/171,
fixture_eval 18/18. (`ruff` still not installed in `tsmc`.)

**Next.** Submit the 18-cell sweep. Score both checkpoints, report the 5-bin mean and the
>=3cm figure side by side, and treat seed as the experimental unit when comparing arms —
v27's episode-level bootstrap CI answers "would more episodes change this", not "would more
seeds".

## 2026-08-27 — dead-code sweep (Tier A) and a silent plotting bug

**Question.** The repo had accumulated single-use experiment code. What is actually
unreferenced, and what only *looks* unreferenced?

**What ran.** An AST scan over every top-level def/class/const in the repo, a module
import graph, an unused-import pass, and `diff -q` of each `slurm/` launcher against the
copies archived under `logs/<sweep>/*/submit_script.sh`. Then all five gates before and
after.

**Gates.** Before: 22/27/41/171/18. After: **25/27/41/170/18**, all green. The two deltas
are accounted for, not lost tests: `static` +4 −1 (four new plotting checks, minus the
`option_graph._port_eval` import check) and `test_option_graph` −1 (`cmd_layering` emits
one check per module under `option_graph/`, and `_port_eval.py` is gone).

**Deleted.**

- `.cleanup-backup/` — six files. The four `.bak` copies were **byte-identical** to the
  live `tests/*.py` (measured, `diff -q`); `gates.sh` ran only four gates and omitted
  `contact`; `status.md.orig` was the 120 KB pre-condense draft.
- `tests/probe_region_budget.py` (269 lines) — its own docstring said *"DISPOSABLE —
  delete once the reference budget is chosen. Nothing imports this."* The budget is chosen
  (`option_budget=50`, `h_region=160`).
- `option_graph/_port_eval.py` (10 lines) — the retired `eval_harness` shim. Four call
  sites repointed: `train.py:133,197`, `tests/fixture_eval.py:172`, plus the import-graph
  entry in `test_code.py`. Closes a TODO item.
- `domains/nav/gym_env.py`'s `setup_logger` plus the `logging`/`sys` imports it orphaned.
- `domains/contact_templates.py`'s `within_cone` and `as_cell_set`;
  `option_graph/planner.py`'s `route_cost`; `option_graph/analysis/plots.py`'s
  `plot_rollout`.
- Five genuinely unused imports (`nav/base.py` `dataclass`, `nav/car.py` `jax` and `np`,
  `edge_model.py` `field`, `test_code.py` `json`).
- `test_code.py`'s `[legacy-ref]` escape hatch, which no source line used.

**The bug: `plot_rollout_grid` drew no rollouts.** `_draw_rollout_ax` took
`X, goal, success, dist, midpoints` and used **none of them** — its whole body was the
region-tint `imshow`. So `eval_harness.py:159` wrote `rollouts/grid.png`, captioned
*"worst 8 rollouts (failures + longest horizon)"*, containing only a colour wash. Fixed to
draw walls, the trajectory (green arrived / red failed), start, goal, interface markers,
and a per-panel title. `plot_rollout`, the never-called sibling with the same defect, was
deleted rather than fixed.

This is the **second** bug in this one file from the same cause, and the file already had
a gotcha about it (*"`train.py`'s plotting path had zero test coverage, and it showed"*).
So it got a gate: `static`'s "rollout figure actually draws the rollout" asserts on the
Matplotlib artists, not on a file existing — the broken version wrote a perfectly valid
PNG. **Verified non-vacuous**: replaying the old function body against the same
assertions gives 0 full-length polylines (needs 1), 0 collections (needs >=3), and an
empty title. All three fail.

**Two claims retracted mid-task, both mine, same root cause.** `grep` in this shell is a
wrapper passing `--ignore-files`, so it honours `.gitignore` — and `.gitignore` has
`tests/*`. Every recursive `grep` silently skipped `tests/test_option_graph.py`,
`tests/probe_edges.py` and `tests/summarize_horizon_sweep.py`:

- `geometry.shortest_region_path` was called dead. It is **live**:
  `tests/test_option_graph.py:163,175,177` golden-diffs it against `planner.bfs_route`
  over every pair of every maze (part of the gate), and `tests/probe_edges.py:213,217`
  calls it.
- `domains/nav/base.py` was called dead. It is **live**: `domains/nav/car.py:24` does
  `from .base import DynamicalSystem` and `DubinsCarSystem` inherits it. The import graph
  missed it because the script only followed absolute imports (`node.level == 0`).

Also retracted: `nav/maze.py`'s `bilinear_sample` (the block carries an explicit
`# noqa: F401`) and `fixture_eval.py`'s `_pin_threads` (documented as a deliberate
re-export). Both are intentional façades, not oversights. The AST scan, which walked
`tests/` with `os.walk`, was sound throughout; only the ad-hoc greps were wrong.

**Not touched, deliberately.** `contact/hooks.py`, `metrics.n_rho`/`aulc`,
`records.flatten_episodes`, `push_arrival`'s `theta_target` path, the risk-aware planner
and `replan` — all built ahead of need but named in `docs/TODO.md` as planned. The
`unmask` and `min_progress_cm` knobs are closed science but still cheap; left for a
separate decision.

**Tier D executed the same day — the serializers are unified.**

There were **five** copies, not four: `tests/probe_edges.py` held a fifth, invisible to
the same `.gitignore`-aware grep that produced the two retractions above. All five now
call `records.json_safe`, whose branch order matters and is documented at the definition
(`np.float64` subclasses `float`, so the NaN test must stay ahead of the `np.floating`
test).

**How it was verified, in increasing strength.**

1. **All five old bodies vs the new one on every real payload.** `model.json`,
   `nine_rooms_n8_summary.json`, `metrics.json`, `edges_n8_h50.json`,
   `eval_composition.json`, plus 400 lines each of `records.jsonl` and the calibration
   jsonl: **byte-identical for all five copies on all seven payloads.** The divergence
   was latent, never active — worth recording, because it means no artifact on disk was
   ever wrong.
2. **The exact `write_jsonl` path** (`separators=(",",":")`, no `sort_keys`): 400/400
   lines byte-identical. `OptionRecord.__post_init__` coerces `entry_state`/`exit_state`/
   `target` to `[float(v) for v in ...]`, which is *why* `records._clean` never needed an
   ndarray branch.
3. **`metrics.json` regenerated through the real Hydra CLI twice** — once on the old
   local `_json_safe`, once on `records.json_safe`, identical args
   (`draws=500 seed=0`) — **identical, md5 `e0609b25…`, 38,490 bytes.** This tests the
   import wiring, not just the function. The run also reproduced the Stage 0 headline
   unchanged: R=4.17, `d_point`=+0.1463, MAE 0.1924 marginal / 0.0461 handoff.
4. **A `static` check that `json_safe` is defined in `records.py` and nowhere else**,
   proven non-vacuous by planting a copy in `planner.py` (gate FAILs, naming both files)
   and removing it (gate passes). It excludes `tests/fixture_eval.py`'s `_clean` **by
   path**, with a comment — that one is a tol=0 value normalizer, not a serializer, and
   relying on its parameter name to miss the regex would have been luck.

`edge_model`'s `model.json` was deliberately **not** regenerated: refitting the MLP needs
the original `--hidden/--epochs/--seed/--val-frac`, which are not recorded anywhere, so a
mismatch would have been uninterpretable. The two-run identical-args comparison answers
the actual question without that confound.

**Behaviour deltas, stated rather than buried.** Against the three *weaker* copies the
unified version now succeeds where they raised: `np.ndarray` (1-D and 2-D), nested
ndarray, `np.int64`, `np.float32` NaN, and tuples containing NaN. Against the two
stronger copies (`edge_model`, `metrics`) it is exactly equivalent. `np.bool_` still
raises, in the new version as in all five old ones — left alone, since no writer produces
one and inventing a branch would be speculative.

**Process note.** A `git checkout -- option_graph/planner.py`, used to clean up a planted
test copy, silently reverted the `route_cost` deletion from earlier the same day. Caught
by re-grepping rather than by any gate. **Never use `git checkout --` as an undo on a
file with unrelated uncommitted work** — the deletion had to be redone.

**Gates after Tier D:** `static` 26/26, `geometry` 27/27, `contact` 41/41,
`test_option_graph` 170/170, `fixture_eval` 18/18.

## 2026-08-27 — the contact video shows the task, not just the object

**Question.** The mp4s were hard to read. What is actually missing, and is it a drawing
bug or a data-plumbing one?

**Cause: data plumbing.** `visualize.Snapshot` had ten fields — board, object pose, both
fingertips, contact flags, walls — and **no goal**. `plot_snapshot` therefore could not
draw a goal even in principle, so every video showed an object moving for no visible
reason. The goal was one stack frame away the whole time: `eval_contact.rollout` computes
`dg`, `d0`, `min_dist` and `min_tick` in the same loop that collects the snapshots.

**Five changes, all verified on a real v27 checkpoint** (`push_legacy_s0`, `model_best`):

1. **Goal + tolerance.** `Snapshot` gains `goal_xy` and `arrival_eps_cm`, optional with
   defaults so every prior caller renders exactly as before. Drawn as a star plus a dashed
   ring. `arrival_eps=0.4cm` on a 50cm board is under 1% of frame width, so the ring has a
   floor of `0.02 * board_w`.
2. **Agency, separately from contact.** Fill stays green/grey for touching/not; the EDGE
   now encodes which fingertip the policy drives — heavy solid for driven, dashed for a
   masked one. Under `mask_inactive_finger` the free tip is servo-HELD in the object's
   path, and contact colour cannot express that; it is the confusion that cost v26 an
   iteration (`forbidden_contact` 8% -> 17-19%).
3. **Object trail + closest approach.** The trail is drawn only up to the current tick, so
   the video never reveals the object's future. The red ring marks `argmin` distance —
   E3's statistic, and the one thing a final frame cannot show.
4. **A two-line caption** replacing `tick N`: `d0 -> final`, `nearest @ tick`, termination
   reason, success, distance bin, and which finger is driven. Everything except
   `why`/`success`/`bin` is derived from the snapshots, so a caller passing nothing still
   gets most of it.
5. **`plot_trajectory` wired in.** It already drew start/end poses, per-finger paths and
   contact make/break markers and was **never called**; `render_episode` now writes
   `<ep>_path.png` beside each mp4.

**Two legibility bugs found by looking at the output, not by reasoning about it.**
The legend sat `loc="upper right"` — inside the board at 50x30, covering the region the
object travels through; it is now below the axes. And `_GOAL_COLOR` was `#1f77b4`, which
is **exactly `tab10[0]`**, the colour `plot_trajectory` gives finger L: the goal star was
indistinguishable from finger L's own contact markers. Now deep purple, with a note at the
constant. Object tan / walls black / fingers blue+orange / nearest red / goal purple are
mutually distinguishable.

**What the fixed render immediately shows** on a failing episode (seed 1000048, d0 7.51cm,
ended `horizon` at 8.21cm, nearest 2.73cm at t98): the driven fingertip is pressing the
object's left face, pushing it *away* from a goal that sits behind-left, while the masked
fingertip is parked at (2,17) doing nothing. That is the E3 aiming failure as a picture
rather than a percentile table.

**Gate: nine new checks in `contact` (41 -> 50).** They assert the ARTISTS, not that a
file appeared — the broken renderer wrote a perfectly valid mp4, which is exactly why this
survived so long. Checked: the overlay survives `to_snapshot`; the goal star exists and
sits at the goal; the tolerance ring is unfilled and floored above `arrival_eps`; driven
and held fingertips get different linewidths; `goal_dist` and `nearest_index` agree with
`argmin`; `nearest_index` is `None` when no snapshot carries a goal; and the caption names
the termination reason and the driven finger. Same lesson as `plots._draw_rollout_ax`
earlier today, now applied in both renderers.

**Incidental cleanup.** The distance-bin label list was duplicated in
`save_summary_png` and `_table`; both now call one `_bin_labels`, which `_bin_label` (new,
for the caption) also uses — one definition instead of three.

**Gates:** `static` 26/26, `geometry` 27/27, **`contact` 50/50**,
`test_option_graph` 170/170, `fixture_eval` 18/18.



## v28 RESULTS — 2026-08-27 — push works; the cone was the whole story

**What ran.** Job `42248679`, all 18 cells COMPLETED at 400k (399.6-400.0k). Scored twice:
the pinned same-room benchmark (`logs/eval/v28_42248679_sameroom/`, digest `3ddae0eb3e93`)
and a cross-room benchmark (`logs/eval/v28_42248679_crossroom/`, digest `01cd2e4efd6b`,
pins recorded in that directory's protocol note). 36 evals each.

**The digest moved for the benign reason.** `3ddae0eb3e93` vs v27's group B
`7b7d59c82155`, because `push_range_min_cm` was added to the hashed env kwargs. The 60
initial states are **bit-identical** (checked episode by episode), so v28 and v27 group B
numbers compare directly. This is exactly the two-causes gotcha, resolving the harmless way.

**Result, same-room, final checkpoint, 3 seeds pooled:**

```
arm        0-3    3-6    6-9   9-12    12+    all   >=3cm
full      0.92   0.86   0.81   0.67   0.44   0.74   0.694
noclip    0.94   0.83   0.69   0.69   0.36   0.71   0.646
nomin     0.97   0.83   0.83   0.61   0.50   0.75   0.694
cone      0.83   0.42   0.08   0.06   0.03   0.28   0.146
cross0    0.86   0.53   0.22   0.08   0.00   0.34   0.208
cross50   0.86   0.72   0.53   0.61   0.33   0.61   0.549
v27 base  0.78   0.19   0.08   0.00   0.00   0.21   0.069   <- same 60 episodes
```

**The distance cliff is gone.** 12+cm went 0.00 -> 0.44; overall 0.21 -> 0.74; goals >=3cm
0.069 -> 0.694, a 10x. Seed sd is 0.036-0.059 and `cone`'s range (0.250-0.333) never
overlaps the working arms' (0.633-0.800), so this is not seed noise.

**Attribution, single-factor from `full`:**

| removed | result | worth |
|---|---|---|
| flat slip -> `friction_cone` | 0.74 -> **0.28** | **-0.46** |
| `target_clip=10` | 0.74 -> 0.71 | -0.03 |
| the 3cm goal floor | 0.74 -> 0.75 | **0.00** |

**It was the cone, and essentially nothing else.** The `cone` arm had the clip, the floor
and 400k steps and still landed at 0.28. The v26 decision to derive the tangential budget
from Coulomb friction cost 0.46 on the primary metric; the v27 estimate of 0.126 at 150k
understated it by ~4x, because the two interact with budget (below).

**The cone does not just cost performance, it CAPS the ceiling.** Mean in-training eval by
step bucket:

```
             0-50k  50-100k  100-200k  200-300k  300-400k
full         0.106    0.192     0.475     0.625     0.684
cone         0.073    0.125     0.199     0.277     0.256   <- flat from 200k, then down
cross0       0.050    0.362     0.553     0.655     0.696
```

Extra budget bought the flat arm ~+0.21 over the last half of training and the cone arm
nothing. Reconciling against v27: flat slip went ~0.23-0.29 at 150k -> 0.74 at 400k
(~+0.45 from budget), the cone 0.15-0.21 -> 0.28 (~+0.09). **v27's "nothing converged at
150k" was right, and the reason the 150k arm comparison understated the cone's cost is that
it ranked the arms by convergence speed rather than by asymptote** -- the failure mode
recorded in the gotchas, observed.

**Both preregistered predictions failed.**
1. The 3cm floor did nothing: `full` and `nomin` are identical above 3cm (0.694 vs 0.694)
   and `nomin` is BETTER below it (0.97 vs 0.92). Once the action space works, short goals
   do not crowd out long ones. The floor is retained as harmless, not as a win. **The
   train/eval mismatch was real as a measurement but was not the binding constraint.**
2. `cross0` did not fail -- it reached 0.73 on its own benchmark. The prediction reasoned
   from *current success at 12+cm* without checking *training density at 12+cm* (8% for
   `full`, 100% for `cross0`).

**The failure mode inverted.** `full`: 74% arrived, 12% horizon, 12% contact_lost, 2%
forbidden_contact, retention 0.92. Of the failures **53% come within 1cm** against v27
`base`'s 19%. Push no longer fails by never getting close; it fails by arriving and not
settling. This is the first evidence that would justify revisiting `require_settled`, which
has been ruled out since v26 -- E3's diagnosis was correct for the policy that existed then.

**The critic is calibrated for the first time.** Q-minus-realized went +5.06 (v27 `base`)
-> **+0.10** (`full`), with `noclip` at -0.36 and `nomin` -0.05. `cone` +2.53 and `cross0`
+3.08. **This closes TODO item 9:** the gap was a symptom of a policy that could not deliver
the value it predicted, not a separate defect. `target_clip` contributes little on its own.

**Cross-room, 32 episodes, goals 12-34cm (final / best):**

```
cross50  0.667 / 0.729      cross0  0.615 / 0.729
noclip   0.604 / 0.490      full    0.531 / 0.458      nomin  0.542 / 0.438
cone     0.073 / 0.052
```

**Transfer is asymmetric and the varied task is the better teacher.** `full` reaches 0.53
cross-room having never trained there; `cross0` manages only 0.34 back on same-room, because
it only ever learned to push east or west (measured: 100% of its pushes are +/-x, against
34% north / 32% south / 18% east / 17% west for `full`). Best single cell anywhere is
`full_s2` at **0.844 cross-room**, beating the specialists on their own task.

**Also measured: the portal is not an obstacle.** Over 400 cross-room resets, the straight
object->goal path is blocked by the wall in **0%** of episodes; the whole object clears the
opening in 89.2%. The opening is y in [5,25] of a 30cm board, `wall_margin=6` confines
objects to y in [6,24], and `_sample_goal_in_push_cone` explicitly requires the ray to pass
through the portal. **"Crosses a room" currently means "a long straight push that passes a
doorway", not a traversal.** Any claim about portal crossing needs a harder board.

**Decision: `friction_cone` is DEPRECATED**, ablation arm only, never the default. NOT
deleted -- the v26/v27 archived cells trained with it and must stay replayable.

**Next.** Recontact at 400k under the same action-space fix; a board where the portal is a
real constraint; and `require_settled`, reopened by the near-miss failure profile.

## v29 — 2026-08-27 — costing the scaffolds; two came back negative for free

**Question.** v28 got push to 0.739 by removing the friction cone. But the setup contains
at least nine things that make the task easier than "push an object to a pose", and none
had been costed. Which are load-bearing?

**The nine, all confirmed in code.**

| # | scaffold | where | status |
|---|---|---|---|
| 1 | gap assist: cannot command retreat faster than the object recedes | `_contact_frame_velocity` | v29 arm |
| 2 | idle finger zeroed | `mask_inactive_finger` | v29 arm |
| 3 | action read as push/slide in the contacted face's frame | `action_interface` | v29 arm |
| 4 | finger spawns ALREADY touching the correct face | `clearance = finger_radius - 0.02` | not a knob |
| 5 | goals confined to a 30deg reachable cone | `push_cone_deg` | **costed, ~0.02** |
| 6 | no orientation requirement | `theta_target` never passed | needs goal-space work |
| 7 | object always spawns at heading 0deg (measured 300/300) | `_place_object_for_push` | v29 arm |
| 8 | rotational damping tuned 6x | `angular_drag_arm_cm` | **costed, 0.033** |
| 9 | portal too open / rooms too small | board geometry | **costed, 0.42** |

**PHASE 0 — three costed at ZERO training cost by replaying v28 checkpoints**
(`logs/eval/v29_phase0/`, 9 evals, ~10 min):

```
damping 6.00 -> 3.12   0.739 -> 0.706   MATCHED (identical 60 episodes)
goal cone 30 -> 90deg  0.739 -> 0.722   (different episodes -- absolute, not matched)
portal 20 -> 10cm      0.615 -> 0.198   3x COLLAPSE
```

Paired per seed, the damping delta is -0.017 / -0.083 / 0.000, mean **-0.033**. **Two of
the three came back negative**, which cut `widecone` from the sweep entirely and reframed
the round: the physics-fairness concerns are NOT propping up the result; the GEOMETRY is.

**The damping value is unphysical anyway, and that is derivable.** `angular_drag_arm_cm` is
the lever arm in `tau = mu*m*g*L`. `L` is not free: for a body sliding on a plane it is the
pressure-weighted mean radius of the contact patch. For this 10x6cm object:

```
uniform pressure (standard assumption)   L = 3.12 cm
uniform disc of the same half-diagonal   L = 3.89 cm
ALL load on the two farthest corners     L = 5.83 cm   <- HARD PHYSICAL CEILING
value shipped in the code                L = 6.00 cm   <- above the ceiling
```

No pressure distribution can produce 6.00. It is 1.92x the standard value and 0.17cm past
the half-diagonal. The comment claimed "measured, not derived"; what was measured is that
1.0 spun the object out and 6.0 did not. Second friction-model mistake in this repo, the
mirror of v26's: that one was OVER-derived (a second friction model over the solver's),
this one UNDER-derived (a tuned constant where a closed form exists).

**A claim of mine was wrong and the data killed it.** I wrote that rotation "dominated
every failure mode". Tick-tracing `full_s2` over all 60 episodes says otherwise:

```
group          n   max |rotation| median   p90
arrived       48        4.8deg          10.7deg
contact_lost   5        7.2deg           7.5deg
horizon        6       13.1deg          13.8deg
```

Median 4.9deg over a whole episode; 9 of 60 ever exceed 10deg. The -81deg spin that seeded
that story was the `angular_drag_arm_cm=1.0` BUG, fixed long ago, and the v21
steering-vs-retention tension was a property of the raw action space, replaced in v25.
Both were carried forward as if still live. **Rotation is a non-issue here largely BECAUSE
of scaffolds 7 and 8** — which is why orientation goals (6) cost so little today.

**The geometry, quantified.**

```
room 25 x 30 cm; usable object-centre box 13 x 18 (wall_margin 6); object 10 x 6
  usable width / object length = 1.3     max same-room goal = 22.2cm = 2.2 object lengths
portal 20cm of a 30cm wall = 67% OPEN; 2.0x object length; 5.0cm clearance each side
  objects spawn at y in [6,24] -- entirely INSIDE the gap [5,25]
  straight object->goal path blocked by the wall in 0 of 400 cross-room resets
```

The board is simultaneously **too small for long pushes** (which is why 12+cm same-room
goals are 8% of episodes) and **too open for crossing to mean anything**. "Crosses a room"
currently means "a long straight push that passes a doorway". A 10cm gap is exactly one
object length, so a broadside crossing has zero clearance -- hence the 3x collapse above.

**Built.**

- **`gap_assist`** (INTERFACE key). The clamp only fires when the object recedes FASTER
  than the finger pushes, and then drags the finger inward to chase it -- a 0.1 push against
  a 12cm/s recession is applied as 12cm/s. A stationary object never triggers it, so the
  first version of the gate test asserted on a case where the branch provably cannot fire.
  Rewritten to test it where it bites, with an explicit "the branch fired" check.
- **`object_theta_spread_deg`** (TASK key). More than a flag: the finger's face offset, the
  face normal, and the coned goal direction all had the axis-aligned assumption baked in and
  all now rotate with the object. Raises if set without `push_cone_deg`, since the historical
  sampler picks faces from an axis-aligned table. `obs()` already carries `IDX_OBJ_HEADING`
  first, so no observation change was needed; the rotated bounding box reaches at most the
  half-diagonal 5.83cm, inside `wall_margin=6.0`, so no rotation can clip a wall.
- **v29 launcher**, 24 cells = `{nogapassist, unmask, rawact, randtheta, physdamp, hardmode,
  narrowgap, bigroom} x seed {0,1,2}` at 400k. Baselines REUSED not retrained: v28's `full`
  is Family A's control, `cross0` is Family B's.

**Cleanup finished this session.**

- **`geometry.shortest_region_path` is now a re-export of `planner.bfs_route`.** One
  implementation, so they cannot drift; lazy import so `domains/` still loads without the
  core. **This made the old gate check vacuous** (it golden-diffed the two against each
  other), so it was replaced with an EXHAUSTIVE simple-path oracle -- a different algorithm
  that checks every pair's route is a valid walk with correct endpoints and optimal hop
  count. Strictly stronger than what it replaced.
- **`base.yaml` audited against the frozen Stage 0 weights.** The TODO flagged `step_pen`;
  in fact **eleven keys differ, five substantively** (`step_pen` 0.00 vs 0.01, `wall_margin`
  0.25 vs 0.0, `horizon` 160 vs 200, `eval_horizon` 640 vs 600, `gamma` 0.99375 vs 0.995).
  So `base.yaml` does not reproduce the weights every published Stage 0 number came from.
  **Deliberately NOT reconciled** -- the values are load-bearing for closed results and F4
  must pick a side key by key. The warning now sits in `base.yaml` next to `step_pen`.
- Already landed before this session, by the in-progress refactor: `_port_eval.py` deleted,
  `json_safe` consolidated into `records.py`, `_LABELS_BY_MAZE` and the hardcoded `vmax=9`
  and the reward-decomposition panel gone, `_load_run_cfg` de-duplicated.

**Gates:** static 26/26, geometry 27/27, **contact 60/60**, test_option_graph **172/172**,
fixture_eval 18/18.

**Verification before submission.** All 24 cells expanded from the launcher FILE (a retyped
copy had a typo the file did not). All 8 arms smoke-trained. Every arm verified
BEHAVIOURALLY. That check caught a bug in my own probe twice -- it let
`PlanarFingertipParams` fall back to its default 80x60 board, which made `narrowgap` look
like it had a broken goal distribution. Re-measured with the board pinned: `narrowgap` has
0.2% sampler fallback and a 22.5cm median against `cross0`'s 21.7cm, so it IS a clean
single-factor change. `bigroom` is not: median 42.6cm vs 21.7cm, so it moves room size AND
push distance together -- unavoidable, but state it.

**Next.** Submit v29. `hardmode` is the number that matters: near 0.7 and the scaffolds were
scaffolding, near 0.2 and v28's result is mostly the setup. My guess is 0.35-0.5.

## v29 RESULTS — 2026-08-28 — the sweep had already run; four of five scaffolds are free

**Correction to the handoff docs.** `status.md` and `docs/TODO.md` both said the v29 sweep
was built and NOT SUBMITTED. It was submitted and finished: job **42300917**, 24 cells, all
24 at ~400k steps with `model.zip` on disk, launched 2026-08-27 14:36 — the same minute as
commit `775bf98`. The docs were written before submission and never updated. Trained but
unscored is the state that looks identical to not-run in a handoff doc, and it lasted a day.

**Scored three ways, 108 evals.**
`logs/eval/v29_combined_sameroom/` (pass a, 78 evals: all 24 v29 cells plus all 18 v28
cells RE-SCORED under identical pins), `logs/eval/v29_42300917_ownsettings/` (passes b and
c, 54 evals), `logs/eval/v29_floor/` (2 evals, the floor). Each directory carries a
`PROTOCOL.md` next to its numbers.

**The digest moved and the reason was checked, not guessed.** `3ddae0eb3e93` ->
`daee708c3fa6`, caused by ADDING `object_theta_spread_deg` and `angular_drag_arm_cm` to the
hashed kwargs. All 60 initial goal distances are identical episode by episode and
`full_s0/model` re-scores at exactly 0.750 against the stored v28 json, so v27/v28/v29
numbers compare directly. Re-scoring v28 rather than quoting it means every number in the
pass-(a) directory shares one digest by construction.

### PASS (a) — common same-room benchmark, digest `daee708c3fa6`, final checkpoint

```
arm            0-3   3-6   6-9  9-12   12+    all     sd   >=3cm  reten   Qgap   len
nogapassist   0.94  0.97  0.89  0.81  0.50  0.822  0.048  0.792   0.94  -0.57    57
physdamp      0.94  0.92  0.86  0.67  0.56  0.789  0.035  0.750   0.93  -0.09    61
unmask        0.97  0.94  0.81  0.61  0.44  0.756  0.084  0.701   0.94  -0.11    59
nomin (v28)   0.97  0.83  0.83  0.61  0.50  0.750  0.073  0.694   0.93  -0.05    69
full  (v28)   0.92  0.86  0.81  0.67  0.44  0.739  0.067  0.694   0.92  +0.10    65
noclip (v28)  0.94  0.83  0.69  0.69  0.36  0.706  0.067  0.646   0.90  -0.36    71
randtheta     0.92  0.92  0.64  0.58  0.42  0.694  0.038  0.639   0.92  +0.22    72
cross50 (v28) 0.86  0.72  0.53  0.61  0.33  0.611  0.048  0.549   0.92  +0.71    91
cross0  (v28) 0.86  0.53  0.22  0.08  0.00  0.339  0.059  0.208   0.90  +3.08   110
narrowgap     0.89  0.36  0.19  0.08  0.06  0.317  0.060  0.174   0.93  +3.33   128
cone   (v28)  0.83  0.42  0.08  0.06  0.03  0.283  0.044  0.146   0.98  +2.53   152
rawact        0.72  0.25  0.06  0.06  0.00  0.217  0.029  0.090   0.62  +1.19    50
hardmode      0.81  0.08  0.00  0.03  0.00  0.183  0.017  0.028   0.60  +2.57    25
UNTRAINED cf  0.75  0.00  0.00  0.00  0.00  0.150      -  0.000   0.98  -1.26   146
UNTRAINED fv  0.25  0.08  0.00  0.00  0.00  0.067      -  0.021   0.38  -0.57    15
```

**THE FLOOR EXISTS NOW** (TODO Immediate #10, closed). A zero-gradient-step network scores
**0.150 / 0.000** (contact_frame) and **0.067 / 0.021** (finger_velocity) on all / >=3cm.
Two consequences. (1) The 5-bin mean has a 0.150 floor from the 0-3cm bin alone — an
untrained contact_frame policy scores **0.75 in that bin** while scoring 0.00 in every other
bin, because gap_assist plus the contact frame keep it touching and pushing (retention 0.98
doing nothing useful). **The >=3cm column is the primary metric; the 5-bin mean is not.**
(2) `hardmode` at 0.028 >=3cm against a 0.021 floor is **indistinguishable from untrained**.

### The preregistered `hardmode` prediction failed, and the arm cannot answer its question

Guess was 0.35-0.5; measured **0.183 all / 0.028 >=3cm**, i.e. the floor. But `hardmode`
bundles four changes and `rawact` alone gives 0.217/0.090, so **the collapse is fully
accounted for by the action interface** — the one scaffold already known to matter since
v25. Removing the other three costs nothing (below). `hardmode` is confounded by
construction and should not be read as "the scaffolds were the result".

### Single-factor attribution: FOUR of five scaffolds are free, and two are NEGATIVE

Paired on the same 60 episodes, per seed, against `full`:

```
                per-seed delta        mean     McNemar (episode-level)
nogapassist   +0.100 +0.100 +0.050  +0.083   25 win / 10 loss   p = 0.017
physdamp      +0.000 +0.150 +0.000  +0.050   27 win / 18 loss   p = 0.233
unmask        -0.083 +0.100 +0.033  +0.017   26 win / 23 loss   p = 0.775
randtheta     -0.033 -0.017 -0.083  -0.044   21 win / 29 loss   p = 0.322
rawact                                -0.522
```

**Removing the gap assist made push BETTER**, +0.083 paired, 3/3 seeds positive, p = 0.017
episode-level. It is the largest single-factor effect in the round and it points the wrong
way for a scaffold. Mechanism is visible in the failure histogram: `nogapassist` has the
lowest failure count of any cell scored (32 of 180) and the lowest `contact_lost` (9%
against `full`'s 12%). Inferred, not yet tick-traced: the assist drags the finger inward to
chase a receding object, which keeps contact but commits the finger to a face the policy
would rather leave.

**Caveat that must travel with that number: `model_best` disagrees.** On the best
checkpoint `nogapassist` is 0.711 against `full`'s 0.739 — the ordering inverts. By the
repo's own rule (`model_best` is the max of a 16-episode eval, so a lucky draw as often as
a peak; if the two disagree the runs have not converged), the gap-assist result is
**PLAUSIBLE, NOT CONFIRMED**, and 400k is not enough to settle it. `physdamp` holds under
both (0.789 final / 0.767 best), so that one is solid.

**`physdamp` is now the default-worthy value.** The physically correct 3.12cm lever arm
costs nothing and may help: +0.050 paired here, and Phase 0's frozen-checkpoint replay said
-0.033. Those two disagree in sign, which is the expected difference between *replaying* a
policy trained at 6.00 under 3.12 (a transfer penalty) and *training* at 3.12. Retraining
recovers the loss and then some. **The unphysical constant was never buying performance.**

### PASS (b) and (c) — own settings, and the baseline replayed under them

```
arm        own    baseline-transferred    read
randtheta  0.717  full -> 0.578           +0.139 for training on it; task NOT harder
physdamp   0.733  full -> 0.706           +0.027; confirms pass (a)
hardmode   0.228  full -> 0.589           full BEATS hardmode on hardmode's own task by 2.6x
narrowgap  0.344  cross0 -> 0.198         +0.146 for training on it (cross-room bins)
bigroom    0.354  no control possible     standalone only
```

**The transfer control kills the "random heading is free" reading and replaces it with a
better one.** On pass (a) `randtheta` looks slightly worse than `full` (0.694 vs 0.739).
On its OWN task it scores 0.717 while `full` transferred in scores 0.578. So a 90deg
heading spread is not a harder task that the policy barely survives — it is a task the
baseline is measurably worse at, and training on it recovers the gap. Same shape as v27's
`place` lesson, opposite conclusion: pass (c) manufactured nothing here, it *rescued* an
arm that pass (a) alone would have written off.

**`hardmode` fails its own task worse than the baseline that never saw it** (0.228 against
`full`'s 0.589). A policy trained 400k steps on a distribution is beaten 2.6x by one that
never trained there. That is not a hard task, it is a **broken learning setup** — consistent
with `rawact`, and with `hardmode`'s in-training eval buckets going 0.072 / 0.065 / 0.126 /
0.128 / **0.096**: declining at the cap, 1/3 seeds rising, against 3/3 for every working
arm.

### Failure modes split cleanly into three families

```
arm            terminations (pass a, final)                      ep_len  mode
nogapassist    arrived 82%  contact_lost  9%  horizon  8%           57   healthy
full           arrived 74%  horizon 12%  contact_lost 12%           65   healthy
unmask         arrived 76%  horizon  9%  FORBIDDEN 8%  c_lost 7%    59   own-finger collisions
rawact         CONTACT_LOST 66%  arrived 22%  horizon 12%           50   cannot hold contact
hardmode       CONTACT_LOST 69%  arrived 18%  FORBIDDEN 11%         25   cannot hold contact
narrowgap      HORIZON 53%  arrived 32%  contact_lost 10%          128   runs out of clock
cone (v28)     HORIZON 68%  arrived 28%  contact_lost  4%          152   runs out of clock
```

1. **Cannot hold contact** (`rawact`, `hardmode`): `contact_lost` 66-69%, episodes 25-50
   ticks, retention 0.60-0.62 against 0.92-0.94 for every working arm. Dies early. This is
   the v21 steering-vs-retention tension, exactly as `contact_frame` was built to fix.
2. **Runs out of clock** (`narrowgap`, `cone`, `cross0`): `horizon` 42-68%, episodes
   128-158 ticks, retention 0.90-0.98 — holds contact fine, never arrives. Median closest
   approach 3.0-3.3cm same-room and **11.6cm** for `narrowgap` on its own cross-room task,
   so it is never-got-close, not fails-to-settle.
3. **Arrives and does not settle** (`full`, `nogapassist`, `physdamp`, `randtheta`): 42-69%
   of the few failures come within 1cm, median closest approach 0.5-2.1cm.
   `require_settled` is a live question for family 3 ONLY, and is beside the point for 1
   and 2 — the same E3 split, re-derived per arm.

**`unmask` is a wash on success and worse on safety.** 0.756 vs `full`'s 0.739 (p = 0.775,
one seed negative), but `forbidden_contact` goes 2% -> 8% final and 2% -> 11% best. v27
measured 26.7-31.7% against a 19% prereg bar; the 8-11% here is much lower, and the
difference is `disengaged_away_deg=60`, which every v29 cell trains with and v27's `unmask`
arm did not. **Keep masking** stands, on the safety column rather than the success column.

### The geometry result survived and is the one worth acting on

`narrowgap` 0.344 on its own task against `cross0` transferred in at 0.198 — training helps,
but the ceiling is low and the mode is `horizon` 59% with median closest approach 11.6cm.
Halving the portal is still the largest task-side effect anywhere in v28/v29, and unlike
`hardmode` it is a single clean factor (0.2% sampler fallback, 22.5cm median goal against
`cross0`'s 21.7cm).

**`bigroom` cannot be scored on a common benchmark AT ALL, and this is a new hard
constraint.** SB3's `check_for_correct_spaces` compares the saved `Dict` observation space
including the goal `Box` bounds, which are `[board_w, board_h]`. So a 90x60 checkpoint
raises `ValueError` against a 50x30 env and vice versa — all 6 pass-(c) transfer evals and
all 6 pass-(a) evals for that arm failed. The Phase 0 gotcha said a board change "confounds
transfer and must be trained"; it is stronger than that — **it is not loadable**, so board
size can never be an arm in a sweep that shares a benchmark. Its standalone 0.354 stands
next to a median goal of 42.6cm against `cross0`'s 21.7cm, so it moves room size and push
distance together and is uninterpretable as a single factor.

**Gates:** static 26/26, geometry 27/27, contact 60/60, test_option_graph 172/172,
fixture_eval 18/18. New: `tools/score_v29_bc.py`, `tools/summarize_v29.py`,
`tools/make_untrained_ckpt.py`, `slurm/score_v29_bc.sh`.

## v31 — the push/recontact spec: target sets, mode guards, HER pool filtering (2026-08-28)

**Question.** An audit of the code against the memo's own definitions of push and
recontact: does what we train match what the graph needs? Nine things did not. This
entry records the eight that were fixed and the one that was deliberately deferred.

**Found by auditing, confirmed by measuring:**

1. `portal_arrival` was DEAD in exactly the configuration that sets it. `ContactEnv.step`
   built `theta_kw` with the portal interface and then REASSIGNED the dict inside the
   `pose_goal` branch, dropping `iface`. Measured: set on 30/30 resets, discarded on all
   30. `_her_arrived` never consulted it either. The portal-region contrast would have run
   as *pose only, twice* — the v16 dropped-variable failure, third occurrence.
2. **A 10x6 object passes a 10cm portal at only 31.2% of orientations.** The admissible
   band is |theta| <= 28.1deg (worst-case y-extent 11.66cm at 59deg). So a portal goal with
   a freely-drawn orientation is ~69% geometrically impossible — a ceiling no training run
   can move. Both ends of a crossing edge are now drawn from the band; that is edge
   FEASIBILITY (sec 6.4), a graph property, not something a policy should discover.
   Verified 300/300 portal goals fit; with the band removed, 73/120.
3. `push_guard` never checked the contact FACE, though Eq 7 makes it an edge parameter and
   xi shows it to the policy. A finger that rounds a corner onto another face still passed
   "is touching" — the option violates its own edge and is scored a success. This is also
   the mechanism behind v30's reachability surprise (predicted ceiling 0.483, measured
   0.507-0.514).
4. `recontact_guard` enforced NOTHING about contact mode — only `off_board`/`force_limit`.
   Pivot and pinch had no guard at all.
5. `_face_idx` was never set for push, so xi's face block was the constant 0 on every push
   episode.
6. Interfaces were 4/2/8 fixed points, not the target SET sec 6.1 asks for.
7. Push's active finger spawned at the exact face CENTRE every time — measured max
   along-face offset 0.0000cm over 400 resets, against a spec of "a random point along the
   face". (Not yet fixed; see below.)
8. The retracted finger's surface gap ran 0.7-12.9cm (median 7.4) against a spec of 4-8cm.
   (Not yet fixed.)

**HER goal-pool filtering — the change with the best measured payoff.** `her_settled`
applies the settle requirement to the SCORED PAIR: draw a goal, then reject it. At the
measured settle rate that discards most of every batch. `her_valid_filter` applies the same
constraint to the CANDIDATE POOL: only draw from settled, guard-valid ticks. Measured on
the spec config: **16.3% of ticks are settled+guard-valid, but 74.5% of future windows
contain at least one** — 4.6x the retained signal for the same constraint, and no batch-size
cost at all. A window with none falls back to no relabel; reaching backwards for a valid
tick would break the future-causality HER relies on. It also aligns HER's implicit goal
distribution with the option's real target set: places the object came to rest, reached
without breaking the contact mode.

**Design finding, recorded because it constrains the architecture.** "Parameterize the
policy by its init and terminal node" cannot be done the obvious way. Eq 18 splits
`pi(a | o(s), rho(g), xi)`; the terminal node is what `rho(g)` MEANS, and HER rewrites
`rho(g)` within an episode. A target-node label in xi would disagree with the goal on ~80%
of every relabeled batch — the v18 bug, in the one block that is supposed to be immune to
it. So xi carries the SOURCE node's interface class and the edge's face/finger/template;
the terminal node lives in the goal, where HER can rewrite it consistently.

**Measured before launch, so they are not open questions.** Spawning into pinch/pivot does
NOT kick the object: 0.0000cm of object motion on the first tick, max over 400 resets.
Contact flags read 0 at reset for EVERY mode including push — pymunk populates them only
after a step, which had already produced one false alarm this project.

**Deferred, deliberately.** Fingertip positions are object-RELATIVE but world-ORIENTED,
while `_wall_distances` already ray-casts along the object's own axes: the observation is
internally inconsistent, and rotation equivariance has to be learned from data. Fixing it
must move the ACTION frame too (object-frame observations with world-frame raw actions make
the policy undo a rotation it was never told about), and it strands every checkpoint. Its
own change. Also deferred: a brake action — the finger servo already brakes at a=0, but
note that under `contact_frame`, `push = 0.5*(a+1)`, so "stop pushing" sits at a=-1, the
edge of the range where a tanh-squashed Gaussian has vanishing density. That is a real cost
specific to `require_settled`; measure before adding a dimension.

**Gate.** `contact` 113 -> 132 checks. Four mutation tests, each producing a clean,
informative failure: pinch drawn as a torque couple, HER ignoring the valid mask, portal
orientation unconstrained (120 -> 73/120 fitting), guard dropping the face check.

**Submitted** as jobs `42617855` (push spec) and `42617867` (recontact), 2026-08-28 16:05.

---

## v31 RESULTS — the bundle went to zero; the control is the only thing that worked (2026-08-29)

Both sweeps COMPLETED, 18 cells, ~126 GPU-hours. Final / best / last-25% mean, from each
cell's own 16-episode diag eval (NOT the stratified benchmark — enough to read 0.000, not
enough for an arm comparison):

```
job 42617855  push spec  1.2M, 8.0-9.4h/cell
  spec          s0 0.000/0.062/0.002   s1 0.000/0.062/0.005   s2 0.000/0.000/0.000
  spec_raw      s0 0.000/0.062/0.001   s1 0.000/0.062/0.000   s2 0.000/0.000/0.000
  spec_settled  s0 0.000/0.062/0.003   s1 0.000/0.062/0.002   s2 0.062/0.125/0.007

job 42617867  recontact  1.0M, 6.3-7.1h/cell
  recon_base    s0 0.938/1.000/0.906   s1 1.000/1.000/0.941   s2 0.875/1.000/0.935
  recon_goal    s0 0.000/0.062/0.003   s1 0.000/0.062/0.001   s2 0.000/0.125/0.005
  recon_full    s0 0.000/0.125/0.020   s1 0.000/0.188/0.051   s2 0.000/0.125/0.012
```

**The one positive result.** `recon_base` at 0.906-0.941 on 3/3 seeds is the rerun
`docs/TODO.md` had asked for since v23. It confirms recontact survived every interface and
slip change since, and it is the only v31 number comparable to history.

**Push diagnosis, tick-traced on `spec_s0` over 60 episodes.**

```
terminations   wrong_face 43/60 (72%)  forbidden 9  contact_lost 4  horizon 3  arrived 1
median episode length 12 ticks   (66 with the guard off)
counterfactual replay, SAME checkpoint:
  guard_face=TRUE   1/60   median closest approach 6.38cm
  guard_face=FALSE  2/60   median closest approach 4.84cm
  + no theta req    3/60   median closest approach 4.84cm
```

`wrong_face` terminated 72% of episodes at 12 ticks, which is the most likely reason nothing
learned. **But that is INFERRED:** turning the guard off recovers almost nothing on an
already-broken policy, so the replay bounds the SCORING artifact, not the TRAINING one. The
clean test is one retrain with `guard_face=false`.

The guard was correct on its own terms — Eq 7 makes the contact face an edge parameter, so a
finger on another face has left the edge. What it missed is that **v30 had already measured
the forbidden behaviour being used**: policies scored 0.507-0.514 where a "behind the face is
unreachable" derivation predicted 0.483, precisely because a finger sliding ALONG a face
rounds a corner without tripping the 4cm `contact_lost` guard.

**The preregistered worry was wrong about which term failed.** The concern was orientation
(push rotates a 1.8deg median, and the goal window was +/-45deg). Measured: 34/60 episodes
reach |dtheta| <= 22.5deg at some tick; only **1/60** ever reach position < 0.4cm; removing
the orientation requirement entirely moves success 2/60 -> 3/60. This is failure family 2
(never got close), not family 3 (arrives, no settle).

**Recontact-Gamma diagnosis, `recon_goal_s0` over 60 episodes.** Terminations are `horizon`
47 and `object_disturbed` 13, so the new `object_still` guard is NOT the cause. The 4-way
conjunction is what never fires: L within its 0.3cm anchor tolerance **1/60**, R within
2.0cm **2/60**, both touch flags matching **4/60**. HER relabels the whole 6-vector, so
relabeled goals are achieved by construction — which is exactly why the buffer looked
healthy while the real task stayed unreachable. Note the horizon is 100 ticks, sized when
ONE fingertip had to be placed; two now do.

**The methodological result, and it is the important one.** Every v31 change was
individually justified by a measurement and the gate grew 60 -> 132 checks with four
mutation tests. The bundle still went to zero on 9 of 9 push cells, and a 3-arm design
cannot attribute across ~8 factors. This is `hardmode`'s v29 mistake repeated at larger
scale, two screens below where that mistake is written down. **A justified change is not a
free change; count factors against arms before launching.**

**Next.** Bisect, do not re-run: `lean` + `guard_face` alone, 3 seeds, is the cheapest
informative cell. Then decide whether `wrong_face` should terminate at all or only penalize.
For recontact, raise the horizon and/or relax the anchor tolerance before spending more
budget. Regenerate the untrained floor for both new goal spaces — nothing here is anchored.

**Also this round:** v30 `lean` (job 42569985) was cancelled at ~600k of 1.2M on all 6
cells. The 400k snapshots survived, so the budget-matched comparison against v29 that the
sweep was designed for is still recoverable; the 1.2M endpoint is not.

## v32 — v31's arms audited, four bugs, and a reverse curriculum (2026-09-01)

**Question.** v31 put all nine push cells at 0.000 and six of nine recontact cells at ~0.00.
Before re-running anything, check what v31 actually trained, and run the two zero-cost
Phase 0 checks the round skipped.

### Four bugs. Two invalidate v31 arms; two were latent.

**1. `recon_goal`/`recon_full` were scored against a goal they could not reach.**
`ContactEnv.step` calls `score_arrival(target=self._goal_xy[:2])`, and under `gamma_goal`
`_goal_xy` is the 6-vector `[Lx, Ly, Rx, Ry, touchL, touchR]`. So `[:2]` is always finger
**L**'s target, while `recontact_arrival` measures the **active** finger, redrawn every
reset. The intended 6-D test, `_gamma_arrived`, is reached only from `compute_reward`, i.e.
only on HER-relabeled samples.

Measured over 500 resets on the exact `recon_goal` config: a state that **perfectly**
achieves the intended interface goal is scored arrived in **254/500 (50.8%)**. Distance the
env measures at the intended optimum: median 0.00cm, mean 5.03, p90 12.48, max 15.22cm,
against `arrival_eps=0.4`. By mode: push 94/175, pivot 74/154, pinch 86/171.

So on ~49% of episodes the reward, `terminated`, and `is_success` were unsatisfiable by an
optimal policy; on the rest they scored a 2-D subproblem ignoring the second fingertip, both
touch flags, and the per-side tolerance table. Rollout and relabeled rewards used **two
different definitions of success in one buffer**. **~63 GPU-hours of v31 recontact are
uninterpretable, and `docs/TODO.md`'s "raise the horizon / relax ANCHOR_TOL_CM" plan was
tuning a task the env never scored.**

**2. The Gamma observation goes stale on relabel.** `physics.obs` bakes
`(finger_targets - current)/pos_scale` into `"observation"` when `two_finger=True`. Only
`PushRelabelSafeHerReplayBuffer` implements `_patch_observations`; recontact gets
`DonePatchedHerReplayBuffer`, whose version is a documented no-op. The v18 bug, reintroduced
in the one new path. Survivable (SB3's `CombinedExtractor` also feeds the correct
`desired_goal`, which is why `recon_base` still scores 0.94 with the same staleness), but it
is a contradictory input on ~80% of every batch.

**3. `_gamma_tol` is read from whichever episode env 0 is in.** `her_buffer` calls
`compute_reward` via `env_method(..., indices=[0])`; the per-side tolerance is set at reset,
so a whole relabeled batch is scored with one live episode's tolerances, and the 0.3cm anchor
tolerance lands on the wrong finger about half the time.

**4. `portal_goal` without a pose goal emitted a ragged goal space.** With
`theta_tol_deg=null` the env declares a 2-D goal Box and emitted **4-D** on 141/300 crossing
episodes while `achieved_goal` stayed 2-D — so `compute_reward` would have compared a
2-vector against a 4-vector. Never hit, because v31 always set `theta_tol_deg` alongside.
FIXED and gated; the check was verified to fail against the old body.

### Two things v31 believed it was testing and was not

**`portal_arrival` was never enabled.** It defaults false and no v31 launcher set it, so
`_goal_iface` was always None. The round's headline audit fix (the "MERGE, not reassign"
repair that stopped `iface` being dropped) is correct code that **the sweep never executed**,
and `docs/TODO.md` Deferred #4 — "training never tests portal crossing" — is still true.
Cross-room push was therefore "park the object dead-centre in the doorway within 0.4cm",
which is *harder* than the memo's crossing test, not closer to it.

**`her_valid_filter` was false in 6 of 9 push cells.** `submit_spec.sh` sets it false by
default and only `spec_settled` overrides. The change the launcher described as having the
best measured payoff was off in `spec` and `spec_raw`.

### `guard_face`: the exploration cost, measured at zero training cost

The v31 diagnosis ("the guard starved training") was inferred. Direct measurement on the
`spec` config, 300 episodes of a **uniform-random** policy:

```
guard_face=TRUE    wrong_face 233 (78%)   median episode  87 ticks   median max displ 5.21cm
guard_face=FALSE   horizon    271 (90%)   median episode 200 ticks   median max displ 7.24cm
```

So the guard costs 2.3x episode length and 28% of explored object motion. Under a **scripted
straight push** it fires on only 14/200 (7%), and `nearest_face(spawn) == _face_idx` on
400/400 resets — the guard is not misfiring on geometry, it fires on sliding, which v30 had
already measured policies using. Confirms the direction but not the magnitude: a trained
`spec` checkpoint ends at 12 ticks, 7x faster than random, so the final policy was degenerate
rather than merely under-explored.

### Eq 15's curriculum, as written, is INERT on this board

Two problems, found by building it and measuring it.

**The ramp moved the wrong end.** For a crossing edge the goal is pinned at the portal, so
capping the GOAL radius is unsatisfiable — a goal past the portal is never near — and the
constraint silently fell away. Fixed: the cap now bounds where the object may START, which is
Eq 15 literally.

**Even fixed, it barely moves the distribution.** 600 resets/level against a
no-curriculum control (2400 resets):

```
                     no curric   L0(9.25)  L1(13.5)  L2(17.75) L3(22.0)
same-room  median      2.00        2.02      1.94      2.15      1.78
same-room  p90         6.70        6.39      6.42      7.05      6.03
crossing   median      9.68        7.51      9.23      9.25      9.58
crossing   max        13.27        9.25     13.27     13.01     13.10
```

Same-room is flat at every level; crossing binds only at level 0. Two causes. Eq 15 requires
**nested** sets, so a level can only DELETE far starts — it can never make near ones
commoner. And the cone sampler's distance is already bunched near zero (median 2.00cm, p75
4.16), so trimming the tail changes nothing. This is structural, not a coding error.

**The advance gate was also reading the wrong env.** `eval_env` was built with
`curriculum_levels` set and nothing advanced it, so it sat at level 0 forever: the 0.6
threshold was reading the EASIEST distribution, which clears at once and clears again at
every level. Fixed with two envs — one pinned to the full task for reporting, one that tracks
the level and gates advancement (Alg 1 line 13's "held-out LOCAL success").

### `portal_arrival=true` puts the untrained floor at 0.271

Single factor, same seeds, untrained `contact_frame`, 60 stratified episodes:

```
config                            ALL    >=3cm    0-3   3-6   6-9  9-12   12+
srg=0.5  portal_arrival=TRUE     0.333   0.271   0.58  0.17  0.42  0.42  0.08
srg=0.5  portal_arrival=false    0.133   0.021   0.58  0.08  0.00  0.00  0.00
srg=1.0  (no crossings)          0.150   0.000   0.75  0.00  0.00  0.00  0.00
```

The crossing predicate alone moves the >=3cm floor 0.021 -> **0.271**, 13x, and the 6-9 and
9-12 bins (where crossing goals live) go 0.00 -> 0.42. **A random policy crosses the doorway
42% of the time**: `portal_goal` draws both the start heading and the goal from the +/-28.1deg
admissible band so the object begins aligned, and the wall blocks the straight path in 0 of
400 resets. `status.md` said the board was "too open for crossing to mean anything"; this is
the number. **REJECTED for the benchmark** — cells kept as `logs/eval/v32_floor/xing_*.json`.

### The reverse curriculum, and why it deviates from Eq 15 deliberately

`curriculum_mode=band`: draw the GOAL first, then place the object at a distance drawn from
the level's window. Distance becomes something we set rather than a geometric byproduct.
Needs reset-to-arbitrary-state, free in simulation.

**Not nested, and that is the point.** Florensa et al. (2017) grow starts outward from a
fixed goal but keep only those at intermediate difficulty, dropping mastered ones; Backplay
(Resnick et al., 2018) slides a window backward along a demonstration. Both are moving
windows. **The deviation from Eq 15's wording is deliberate and those two papers are the
justification.** ADR (OpenAI 2019) is the counterexample that clarifies it: expansion works
when the sampler follows the boundary, and ours does not.

Windows are fractions of the distance each edge can reach, so no per-edge constants:
`(0.00,0.35) (0.15,0.60) (0.35,0.85) (0.00,1.00)`. Width ~0.35-0.5 so a level restricts
something; consecutive windows overlap by ~0.2 so advancing is not a cliff; **the last level
is the full range** because the benchmark scores every bin and a band final level would be a
train/test mismatch. Measured, 600 resets/level, **0 leaks, every level restricts**:

```
level   window        same-room median   crossing median
  0     0.00-0.35          1.58cm             8.55cm
  1     0.15-0.60          3.28              10.73
  2     0.35-0.85          4.80              11.76
  3     0.00-1.00          2.66               9.53
```

A real ramp on both edge types, ~3x on same-room. Side effect: at full range the reverse
sampler gives a 2.66cm same-room median against the forward sampler's 2.00, which partly
closes the training-vs-benchmark distance gap (benchmark median 7.2cm).

**The control shares the sampler.** `curriculum_mode=band curriculum_levels=null` is the
reverse sampler pinned at the full range with no schedule, so the no-curriculum arm differs
from the ramped arm by the SCHEDULE ALONE. Comparing a band arm against the old forward
sampler would have confounded the two.

### v32 floor, on the exact sweep protocol

`logs/eval/v32_floor/`, with `PROTOCOL.md` beside the numbers. 60 stratified episodes,
reverse sampler at full range, `portal_arrival=false`:

```
cell                     digest         ALL   >=3cm    0-3   3-6   6-9  9-12   12+
pos_contact_frame      3366def8826d   0.117   0.042   0.42  0.17  0.00  0.00  0.00
pos_finger_velocity    3366def8826d   0.000   0.000   0.00  0.00  0.00  0.00  0.00
pose_contact_frame     249434216cd2   0.067   0.042   0.17  0.17  0.00  0.00  0.00
pose_finger_velocity   249434216cd2   0.000   0.000   0.00  0.00  0.00  0.00  0.00
```

**The first version of this floor was WRONG and the digest caught it.** It omitted
`disengaged_away_deg=60`, a TASK key inside the digest, so it scored a different reset
distribution than the sweep trains on (`1a72f6438f34` vs the sweep's `249434216cd2`). The
success numbers were unchanged, which is why only the digest could have found it. The
protocol is now DERIVED from a launched cell's `meta.txt` rather than retyped, and the
verified scoring command lives in `logs/eval/v32_floor/PROTOCOL.md`.

### Gate

`contact` 132 -> **138**. Five new checks: no level exhausts the retry budget; the window
ramp actually moves the distance (with the inert nested medians in the failure message); the
reverse sampler keeps the goal inside the contacted face's cone (worst 29.96deg vs 30); band
with no levels is the full range; band rejects a wrong level count and a missing cone. Plus
the ragged-goal check. All five gates green: 26 / 27 / 138 / 172 / 18.

### Two breaks a code review would not have caught

Found by smoke-testing the real launcher end to end, per the repo's own rule:
`local_env.set_curriculum_level` failed on the `Monitor` wrapper (would have crashed 6 of 12
cells at the first eval), and a 64-draw retry budget left 1 reset in 600 falling back to the
forward sampler at the wrong level (raised to 256, now zero).

### Submitted

Job **43572361**, 12 cells, 600k steps, 2x2 of {curriculum on/off} x {restricted, raw
actions} x 3 seeds. All 12 recorded one `GIT_DIFF_SHA=d93e35ff325287e6` and one launcher
md5, so every cell ran the same tree. Preregistered verdicts are in the launcher header.

**Next.** Score against `logs/eval/v32_floor/`. Then fix the recontact Gamma scoring bug
before any recontact number is quoted, and delete the now-dead nested curriculum path.

---

## v33: pricing the scaffolds, and a face-guard measurement that qualifies v32 (2026-09-01)

**Question.** v32 showed push is learnable with a reverse curriculum. But it is learnable on
a task made easier than the memo's push option in several ways. Which of those scaffolds is
load-bearing, and which is free to remove?

### v32 result, final

Scored with `tools/score_sweep.py --pins` from `logs/sweep_43572361/PINS.txt` into
`logs/eval/v32_final/`, digest **`249434216cd2`**, matching `logs/eval/v32_floor/`.
Goals >=3cm, `model.zip`:

| arm | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| `curric` | 0.604 | **0.750** | 0.667 | **0.674** |
| `base` | 0.562 | 0.604 | 0.583 | 0.583 |
| `raw` | 0.146 | 0.125 | 0.146 | 0.139 |
| `curric_raw` | 0.083 | 0.083 | — | 0.083 |

Untrained floor 0.042 (restricted) / 0.000 (raw). Task 11 (`curric_raw_s2`) died on a wandb
init timeout and never trained; not resubmitted. **The curriculum helps** (3/3 seeds) and
**does not replace the action restriction** (`curric_raw` 0.083 vs `base` 0.583).

### The face-guard measurement, and what it costs the v32 headline

Same two policies, one key flipped, nothing else changed (`logs/eval/v32_faceprobe/`):

| policy | unguarded | `guard_face=adjacent` | `guard_face=true` (strict) |
|---|---|---|---|
| `push_curric_s1` | 0.683 | **0.483**  (wrong_face 13/60, 21.7%) | 0.083  (49/60, 81.7%) |
| `push_base_s1`   | 0.600 | **0.583**  (wrong_face  9/60, 15.0%) | 0.083  (52/60, 86.7%) |

**Face switching is how these policies push.** They leave the contacted face on 82-87% of
episodes, at a median of 12 ticks. A claim made earlier in this session — that the 4.0cm
contact-loss budget bounds face switching — was **wrong**, and is corrected here: that budget
covers only time spent NOT touching, and a finger that keeps contact can slide along the
surface without limit. The governing number is geometric. The finger spawns at the face
CENTRE of a 10x6 object, so the corner is |ly| = 3*5.5/5 = **3.3cm** away, which is ~4 ticks
at 20cm/s. Every episode starts four ticks from a face change.

**The arm ordering inverts under the guard.** Unguarded, `curric` 0.683 > `base` 0.600.
Under `adjacent`, `curric` 0.483 < `base` 0.583 — `base` loses 0.017 and `curric` loses
0.200. Part of the curriculum's v32 advantage is therefore bought with behaviour that
violates the edge label it was executing (Eq 7 makes the face an edge parameter; Eq 40's
chi_push enforces it). **"The curriculum helps on the task as scored" stands. "The curriculum
learns a better push option" does not.** Recorded as a negative result against v32's own
headline, and it is what the v33 `faceguard` arm exists to settle.

Both guarded numbers are ZERO-SHOT — policies that never saw the constraint, evaluated under
it. They are the arm's floor, not a prediction.

### `guard_face` gained an `adjacent` mode, folded into the existing key

`false | true/"strict" | "adjacent"`; `adjacent` forbids only the OPPOSITE face
(`face ^ 1`, `_opposite`). Chosen over `strict` because strict is unlearnable-shaped: v31
took 9/9 push cells to 0.000 on it, and it scores 0.083 here with no training at all.

Implemented as ONE key rather than a `guard_face_mode` companion, deliberately. The env
digest is `sha1` over every env kwarg except the six interface keys, so a NEW kwarg would
have moved the digest of every config in the repo and invalidated `logs/eval/v32_floor` and
all of `logs/eval/v32_final`. Widening the existing key leaves `repr(False)` untouched.
Verified: the v32 protocol still hashes to `249434216cd2` after the change.

### The v33 sweep

18 cells, 6 arms x 3 seeds, 600k steps. Every arm is v32's `curric` with exactly ONE key
changed: `ctl`, `freefinger` (`mask_inactive_finger=false`), `widecone` (`push_cone_deg=90`),
`spread` (`object_theta_spread_deg=90`), `faceguard` (`guard_face=adjacent`), `midaction`
(`finger_velocity` + `restrict_contact_actions=true`). `ctl` is rerun rather than reused
because this tree changed `contact_templates.py` and `gym_env.py`.

Two key classes get two scoring treatments, and this is what makes the sweep readable.
`freefinger` and `midaction` change INTERFACE keys, which sit outside the digest and are read
per cell — same benchmark as `ctl`, directly comparable. `widecone`, `spread` and `faceguard`
change TASK keys and carry their own digests; they are scored on the common tight protocol
automatically, and on their own distribution only as a deliberate follow-up. Verified by
smoke test: `ctl`/`freefinger`/`midaction` all hash to `a2520b17d6e7`, the other three to
three distinct digests.

### Automatic post-sweep media

`slurm/finalize.sh` + `tools/render_best.py`. The last array task to finish auto-submits
finalize, which scores every cell (both checkpoints) then renders the best checkpoint of the
best seed of every arm — 3 hardest arrivals plus 3 of the dominant failure mode — into
`media/<tag>/<arm>/`. `eval_contact.py`'s `informative` pick changed from median arrivals to
HARDEST arrivals: `auto` was leading with 1-2cm goals, which makes a good policy look
trivial.

The benchmark protocol is no longer retyped in the scorer. The launcher writes it to
`logs/sweep_<jobid>/PINS.txt` (atomically, via `mv`, since 18 tasks race) and finalize reads
it. `render_best.py` then asserts the rendered eval reproduces the source eval's digest, so a
video cannot silently be of different episodes than the number it illustrates.

### Gate

`contact` 138 -> **141**. Strict bans every face change and adjacent bans only the opposite
one (all 4 faces x 2 modes, with the finger placed through the env's own `_face_geometry` so
the test cannot drift from the sampler); `guard_face` reads false / true-as-strict /
adjacent; an unknown mode raises. All five green: 26 / 27 / 141 / 172 / 18.

### Launched and verified (2026-09-01 16:00, job 43679344)

18/18 COMPLETED, 4:24-5:09 per cell. Verified at launch rather than assumed, because a
one-factor sweep is worthless if any cell is off by two: all 18 cells present with the right
arm/seed mapping, ONE `GIT_DIFF_SHA=989475b0a6b12832` across all 18, `PINS.txt` byte-identical
to v32's, and every arm diffed against `ctl` **from the recorded `meta.txt`** (not from the
launcher) to confirm exactly one factor moved.

`midaction` shows five token differences and is still one factor: `slip_model` and `slip_limit`
are contact-frame-only parameters with no meaning under `finger_velocity`, and
`restrict_contact_actions` is the defining half of the middle rung.

One false alarm worth recording, because the same trap will recur: `faceguard` and `midaction`
appeared stalled at zero log blocks while the other arms were 14-22 in. They were fine.
Python's stdout to a file is BLOCK-buffered, so `run.out` stays empty until ~4KB accumulates.
**Progress is `progress.csv` and `model_best.zip`, never `run.out` size.**

### The auto-finalize hook was broken, and had never worked on any sweep

All 18 cells finished and `slurm/finalize.sh` never ran. Root cause, measured with a 3-task
probe job rather than reasoned about: the last-task-standing test used
`squeue -o "%A_%a"`, and on this Slurm **`%a` renders as the ACCOUNT name**, not the array
index:

```
raw squeue output:   [43892866_hankyang_lab]
task=2 still=[1]      # can never reach 0
```

So `grep -v "^<jobid>_<taskid>$"` matched nothing, `still` never reached 0, and the block was
dead. It has been dead since it was written: `logs/sweep_{42300917,43572361,43679344}` all
lack `slurm_logs/`, and 621 files / 20MB sit orphaned in `logs/slurm_staging/`.

**This was inherited code that new automation got hung off without checking it worked** — the
repo's own rule (a launcher is the only proof a flag is live) applied to a launcher and was
not followed. Fixed to `-o "%i"`, which does render `<jobid>_<index>`, plus an atomic
`mkdir "${SWEEP_DIR}/.finalized"` guard so two cells finishing in the same instant cannot both
submit finalize. **Untested end to end — the next sweep is the test.** Until then, run
finalize by hand: `sbatch slurm/finalize.sh logs/sweep_<jobid>`.

### A SCORING BUG IN SHARED TOOLING, caught by the preregistered digest check

The first scoring run came out at env digest **`c10067af8f09`**, not the preregistered
`249434216cd2`. That is the declared STOP condition, and it fired correctly.

Cause: `tools/score_sweep.py` built its command as `[*pins, PORTALS, *iface, *group]` with
`PORTALS` a **hardcoded v29-era constant** (`portals=[{x:25.0,y_lo:5.0,y_hi:25.0}]`, a 20cm
doorway). Hydra takes the LAST override, so the constant silently replaced v33's own 10cm
doorway from `PINS.txt`. **`--pins` could not express the portal at all**, despite its help
text promising it "replaces TASK_PINS wholesale". Confirmed by comparing benchmark episodes
directly rather than trusting the digest string: **36 of the 60 initial states differed** from
v32's set, so this was a real change of task, not a cosmetic rehash.

Scope, checked rather than assumed:

- **v33's first scoring: affected.** Cancelled mid-run (job 43892831) and quarantined at
  `logs/eval/v33_WRONG_PORTAL_do_not_use/` with a README, because `score_sweep` skips cells
  whose output already exists and would otherwise have silently reused them.
- **v32 `logs/eval/v32_final/`: NOT affected.** It was scored by a hand-rolled loop, not
  `score_sweep`, and its digest is the correct `249434216cd2`.
- **v25/v26/v28/v29 eval dirs: not affected in the same way** — every one of those sweeps
  trained on the 5-25 portal that the constant happened to hardcode. The one loose end is
  v29's `bigroom` arm, which trained on `x:45,y 10-50` and was scored through `score_sweep`;
  v29's portal arms had a dedicated scorer (`tools/score_v29_bc.py`) with explicit
  WIDE/NARROW/BIGPORT constants, and `bigroom` is already recorded as unscoreable on a shared
  benchmark, so nothing rests on it. Flagged, not silently blessed.

Fixed by folding the portal INTO `TASK_PINS` and deleting the separate append, so `--pins`
is authoritative as documented. Two gates added (`static` 28 -> 30): `TASK_PINS` must contain
a `portals=` token, and the identifier `PORTALS` must not exist in the file — the second one
fails the moment this exact bug regrows.

**The lesson, and it is the same one twice in one session.** The auto-finalize hook was dead
because a `squeue` format was never verified; this was dead because a tool's own help text
was believed instead of its command. Both were found by checking output against a
preregistered expectation. **The digest check is the only reason this sweep is not now a
misleading result** — keep it as a hard stop.

### Rescoring

`sbatch slurm/finalize.sh logs/sweep_43679344 v33` (job 43894601) -> `logs/eval/v33/` and
`media/v33/<arm>/`. `finalize.sh` gained an optional tag argument so the output is `v33`
rather than `v43679344`.

**One reassuring cross-check already in hand.** Rescored by hand under the correct pins,
v33 `ctl_s1` reproduces v32 `curric_s1` to every reported digit — 0.683 all-bins, 0.917
retention, 7.42cm displacement, 0.40cm final, 67 ticks. Same seed, same config, and the
`guard_face` widening was behaviourally inert, exactly as the digest verification predicted.

### v33 RESULT — all 36 evals at digest `249434216cd2`, goals >=3cm

| arm | `model` | `model_best` | vs `ctl` | per-seed (`model`) |
|---|---|---|---|---|
| `ctl` | **0.674** | 0.688 | — | 0.750 / 0.667 / 0.604 |
| `widecone` | 0.604 | 0.562 | **-0.069** | 0.750 / 0.562 / 0.500 |
| `freefinger` | 0.583 | 0.542 | **-0.090** | 0.667 / 0.562 / 0.521 |
| `spread` | 0.562 | 0.597 | **-0.111** | 0.625 / 0.562 / 0.500 |
| `faceguard` | 0.562 | 0.472 | **-0.111** | 0.625 / 0.604 / 0.458 |
| `midaction` | 0.139 | 0.146 | **-0.535** | 0.188 / 0.125 / 0.104 |

**`ctl` reproduced v32 `curric` exactly: 0.674 vs 0.674.** Same seeds, same config, different
job, on a tree that had changed `gym_env.py` and `contact_templates.py`. That is the
replication check the arm was included for, and it passed.

**Verdicts, against the rules written before the sweep.**

- **No scaffold is CHEAP.** Nothing landed within 0.05 of `ctl`. `widecone` came closest at
  -0.069 and still missed the bar. Every one of these four costs something real, so all four
  go in the recorded fidelity-deviation list **with their prices** rather than being quietly
  removed.
- **One scaffold is LOAD-BEARING: the action space.** `midaction` -0.535. And the number is
  exact: **0.139, identical to v32's `raw` arm at 0.139.** So adding
  `restrict_contact_actions=true` on top of `finger_velocity` bought **precisely 0.000**,
  with the curriculum on. v18 measured why and it still holds — 62% of contact breaks are
  TANGENTIAL slides off a corner, which an outward-normal clamp never restricts. The contact
  FRAME is doing the work, not the no-retreat constraint. `midaction_s0` and `_s2` are also
  the only cells in the sweep with an empty distance bin, so they fail Bar 1 outright.
- **Three arms land in the no-verdict band** (0.05-0.15 below `ctl`): `freefinger` -0.090,
  `spread` -0.111, `faceguard` -0.111. Report the numbers, do not re-run hoping they move.
- **`faceguard` clears its own bar decisively.** It was judged against 0.083 (untrained under
  the guard), not against `ctl`. It reached **0.562** — and 0.562 also beats the 0.483
  zero-shot number v32's best policy scored under `adjacent`. **The face constraint is
  learnable, and it costs about 0.111.** That is the single most useful result here: it means
  a push option can respect Eq 7's contact-face parameter at a priced, modest cost, instead
  of the 0.000 that v31's strict form produced.

**What this does to the v32 qualification.** v32's headline was qualified because the arm
ordering inverted under the face guard, so part of the curriculum's advantage looked like it
was bought with edge-label violations. `faceguard` at 0.562 does not erase that, but it
changes the conclusion from "the curriculum may be exploiting a loophole" to "the loophole is
worth ~0.111 and can be closed by training with it closed." The honest statement is now: the
curriculum helps, and a face-respecting push option is available at a known price.

### `faceguard` scored on its OWN distribution — training with the guard is COUNTERPRODUCTIVE

Job 43895909, `logs/eval/v33_faceguard_own/`, digest `96762fdf1de4` (guarded distribution,
deliberately not the tight benchmark). Goals >=3cm:

| arm | trained with the guard? | `>=3cm` | per-seed | `wrong_face` rate |
|---|---|---|---|---|
| `faceguard` | yes | 0.542 | 0.438 / 0.625 / 0.562 | 9.4% |
| `ctl` | **no** | **0.604** | 0.583 / 0.625 / 0.604 | 12.2% |

**The policy that never saw the constraint scores HIGHER under it (+0.062).** Training with
the guard did cut violations (9.4% vs 12.2%), but not nearly enough to pay for the success it
cost. A terminating guard during training removes episodes that were on their way to a
success, and at this strength it still costs more than it teaches — the same mechanism that
took v31's strict form to 0.000, just milder.

**This supersedes the reading written above from the tight-benchmark numbers alone.** The
correct conclusion is not "the face constraint is learnable at ~0.111 by training with it."
It is:

- **The faithful, face-respecting push number is 0.604** — `ctl` scored under
  `guard_face=adjacent`. That is the number to quote for a push option that honours Eq 7's
  contact-face parameter, and it costs **0.070** against the unguarded 0.674, not 0.111.
- **Do not spend cells training under the guard.** `ctl` already satisfies the constraint on
  ~88% of episodes without ever being told about it. Enforce the face at EVAL and report the
  guarded score; that is both cheaper and better.
- v32's best policy scored 0.483 zero-shot under `adjacent` and v33's `ctl` scores 0.604, so
  the guarded number improved between sweeps without anyone optimising for it.

**Still outstanding:** the same own-distribution scoring for `widecone` (`push_cone_deg=90`)
and `spread` (`object_theta_spread_deg=90`). Given how this one turned out, expect the
loosened-training arms to underperform `ctl` on the loosened task too, and budget accordingly.

**Next.** Read `logs/eval/v33_faceguard_own/`. Randomise the face-centre spawn — the centre
puts every episode the same 4 ticks from a face change, and it is now clear the face
constraint is worth respecting. Build the flat baseline; it must inherit all four priced
scaffolds. Then the recontact Gamma scoring bug, and delete the dead nested curriculum path.

## obs v2: one normalizer, one head layout, and two live bugs closed (2026-09-02)

**Question.** The observation vector grew feature-by-feature across v25-v33 and
nobody had ever measured it. Which of its 41 dimensions carry signal, which are
mis-scaled, and is anything simply wrong?

### Measurement first (3 `ctl` seeds x 60 benchmark episodes = 17,506 ticks)

Per-dimension mean / std / min / max / correlation-with-success, plus the full
pairwise correlation matrix, over the concatenated 49-D network input
(`observation` 41 + `achieved_goal` 4 + `desired_goal` 4 — SB3's
`CombinedExtractor` uses `Flatten` for all three keys, so nothing is normalized
anywhere).

**Five findings, all measured:**

1. **Three scale bugs.** Angular velocity had **no divisor at all** (range
   +/-3.27, the only over-range feature) while every neighbouring velocity was
   divided by `v_max`. Peak force was divided by **1000.0** — a fallback reached
   because `force_abort_kgcms2` is `null` in every config — against a measured
   p99 of 284/323, so it only ever occupied [0, 0.5] with typical values
   0.03-0.06. Goal-relative position was divided by the board extent (50) though
   a goal never sits more than one same-room diagonal (22.2cm) away: measured std
   **0.039 / 0.071**, the two weakest signals in the entire vector.
2. **Six dimensions are exactly constant** — the template one-hot (2) and the
   source-interface one-hot (4). **Kept deliberately**, see below.
3. **`xi_act_L` and `xi_act_R` correlate at exactly `-1.000`** — a 2-dim one-hot
   for a binary fact.
4. **The largest scale problem is OUTSIDE `observation`.** The four
   highest-variance dimensions of the whole network input are the goal keys'
   positions — `achieved_goal.x` std **7.79**, `desired_goal.x` 7.55,
   `achieved_goal.y` 4.64, `desired_goal.y` 3.93 — against a median of 0.227 and
   a next-largest of 0.53. That is a **34x** ratio, in raw centimetres. And those
   four dims are absolute board position, which `obs()`'s own docstring says it
   deliberately drops ("so one shared policy sees near-identical inputs for the
   door at x=30 and the door at x=60"). The object-centric design is defeated at
   the goal keys, not inside the observation. **Left for the VecNormalize step;
   NOT fixed here.**
5. **Relative heading is the single strongest predictor of success (0.565)**, far
   ahead of object heading (0.368), wall distance (0.307) and peak force (0.225).
   Third independent confirmation that the ORIENTATION gap, not distance, is
   push's real difficulty axis.

### Two live bugs found in recontact, both confirmed by code trace AND measurement

**A. `obs()` hardcoded `obj_xy` where it meant `achieved_goal`.** The goal tail
was `(target - object_WORLD_position) / scale`, but recontact's goal is a
fingertip target in the **OBJECT's** frame. Measured `goal_rel_x` mean
**-0.492** (= -40/80, the object's board-centre x) over a range of only 0.21 —
the feature was mostly encoding the object's board position.

The correct formula **already existed in this repo**, in push's HER patcher:
`(new_goals[:, :2] - ag[:, :2]) / pos_scale`. `obs()` only agreed with it for
push because push's `achieved_goal[:2]` *is* `obj_xy`. Fixed by passing
`achieved` in; **bit-identical for push**.

It was invisible because recontact pins the object at the board centre, so the
offset is a learnable constant. **It would have broken the moment push handed the
object over anywhere else** — i.e. in composition, in the one template whose
entire job is being composable. This is memo sec 7's row verbatim: "No
unseen-topology transfer -> local policies encode global coordinates or board
identity -> object-centric observations."

**B. The HER goal-tail patch was push-only.** `DonePatchedHerReplayBuffer._patch_observations`
was an empty method body, and `train_contact.py` hands recontact that class. So
recontact's goal-derived tail described the OLD goal on ~80% of every relabeled
batch — **both** the 2-D and the Gamma variant, not just Gamma as the module
docstring claimed.

**The two bugs masked each other**, which is why neither showed: bug A made
recontact's tail near-constant, so a stale value looked like a fresh one. Fixing
A without B would have made recontact **worse** with no visible cause. They land
in one commit, and gate (f) below asserts it.

### What changed

- **`physics.ObsScales`** — the one place any divisor may live. Six named fields
  (`pos`, `goal`, `wall`, `vel`, `omega`, `force`). `.v1()` reproduces the
  historical constants **including both bugs** (`omega=1.0` is the *absence* of a
  divisor; `force=1000.0` was the fallback), so archived checkpoints replay
  bit-identically. `.v2()` fixes all three. Before this the scales sat in three
  scopes and one was a hand-copied duplicate in `her_buffer.py` kept in step by a
  comment.
- **`obs_version=1|2`**, an **INTERFACE key**, not a task key: it changes how the
  policy reads the world, not the reset distribution, reward or horizon.
  **Verified: the v32/v33 digest `249434216cd2` is unchanged**, so
  `logs/eval/v32_floor`, `logs/eval/v32_final` and all of `logs/eval/v33` stay
  valid, no floor re-measurement is needed, and obs v1 vs v2 is an arm scorable
  on ONE benchmark.
- **v2 layout: the state+xi head is 36 dims for EVERY template** (push,
  recontact-2D, recontact-Gamma), asserted by gate (c). `xi` is always emitted at
  `N_XI_V2 = 11`; the active-finger pair collapses to one scalar (finding 3).
  **The constant template and interface one-hots are KEPT** — that is the point,
  not an oversight: a shared head is what lets one template's policy load against
  another's env (Eq 9, sec 6.2's universal-actor ablation, and composition). The
  GOAL block stays template-specific at 4/2/6, because different arity is
  semantically correct. Totals: push **40**, recontact-2D **38**,
  recontact-Gamma **42**.
- **One HER patcher for all templates.** `_patch_observations` moved to the base
  class and is driven by the goal's arity, since that alone identifies the
  template: 2 -> two position deltas; 4 -> two deltas + relative heading;
  6 -> four deltas + the two touch flags. `PushRelabelSafeHerReplayBuffer` now
  carries only the relabel tick lag.
- `pos_scale` survives as a **load-only alias** for `goal_scale`. SB3 bakes
  `replay_buffer_kwargs` into the checkpoint zip and reconstructs the buffer with
  them on `load()`, so dropping the name made every v25-onward push checkpoint
  unloadable — caught by replaying v33 `ctl_s1`, not reasoned about.

### Gates: `contact` 141 -> 159, `static` 30 -> 30 (one rewritten)

New section "obs v2: one head layout, one normalizer, and the frame fix":
(a) `obs()` contains **no bare numeric divisor** and `her_buffer.py` no longer
carries its own position scale; (b) `.v1()` reproduces the historical constants,
bugs included; (c) the state+xi head is 36 for every template; (d)
`obs_version=2` without `rich_obs` raises rather than emitting a short head;
(e) the recontact tail is `(desired - achieved)` with both in the object frame,
**plus a check that v1's tail really did encode board position** so the bug this
replaces cannot be called hypothetical (measured v1 `-0.630` vs v2 `+0.371` on
the same reset); (f) the HER patch reproduces a from-scratch `obs()` for all
three template/goal combinations — it previously covered push only.

`static`'s interface-key check was rewritten from "the expected six keys" to
nine, with a comment recording WHY the three new keys belong outside the digest
and that anything moving the TASK must not join them. It fired correctly on the
first attempt, which is what it exists for.

### Verification

- **All five gates green: 30 / 27 / 159 / 172 / 18.**
- **v1 is bit-identical, verified by replay, not by argument.** v33 `ctl_s1`
  re-scores to 0.683 all-bins / 0.917 retention / 7.42cm displacement / 0.40cm
  final / 67 ticks — every reported digit — and all 60 episodes match the
  archived `logs/eval/v33/1_push_ctl_s1__model.json` on `(d0, success, steps,
  why, q0, final_dist)` with **0 differing episodes**. `recon_base_s1` likewise
  re-scores to 0.983 with all 60 episodes identical.
- **v2 conditioning, measured over 40 untrained episodes per version:**
  observation `|max|` **2.23 -> 1.296** (the angular-velocity over-range is
  gone), goal-tail std **0.080 -> 0.181** (2.25x better). The 34x
  goal-key ratio is **unchanged and still open** — that is the VecNormalize step,
  deliberately not in this commit.
- **Smoke-trained at v2:** push 2,500 steps, and BOTH recontact variants
  (2-D and Gamma with `init_gamma_modes=[free,push,pivot,pinch]`) 2,000 steps,
  which is what exercises the new HER tail patch inside the real training loop.

### Also closed this session: `recon_base` has a benchmark number at last

`docs/TODO.md` Immediate #4 asked for this since v23. Scored on the stratified
60-episode protocol on current code, digest `a78252c0a0a6`, 3 seeds:
**0.967 / 0.983 / 0.983, mean 0.978**, no floored bin, 58-59 of 60 `arrived`.
Better than the 0.906-0.941 diag-eval figure and the v23 protocol's 0.783-0.917.
NOTE this is recontact's own reset distribution, not the push benchmark.

**Next.** VecNormalize on the goal keys only (`norm_obs_keys=["achieved_goal",
"desired_goal"]`) — verified safe because `compute_reward` and `_her_arrived`
run on RAW arrays at `her_buffer.py:130-135`, *before* `_normalize_obs` at
136-137, so `arrival_eps=0.4` keeps meaning 0.4cm and there is no split unit
system. Then the Gamma arrival dispatch, the reward-vector extension, PPO, and
the recontact launcher parity.

## P1-P4: goal-key normalization, the Gamma arrival fix, and a reward vector that can express what a dense push run needs (2026-09-02)

Four items, one commit. **All five gates green: 30 / 27 / 192 / 172 / 18** (`contact`
159 -> 192). Archived checkpoints re-verified bit-identical at the end, not assumed.

### P1 -- VecNormalize on the GOAL KEYS ONLY (`normalize_goal_keys`)

The 34x conditioning problem measured in the obs-v2 entry: `achieved_goal` /
`desired_goal` positions are raw centimetres and are the four highest-variance
dimensions of the whole 49-D network input (std 7.79/7.55/4.64/3.93 against a
median of 0.227), because SB3's `CombinedExtractor` just `Flatten`s every key.

`VecNormalize(norm_obs_keys=["achieved_goal","desired_goal"], norm_reward=False)`.
**`observation` is deliberately excluded**: 22 of its dims are unit-vector pairs
or one-hots, and whitening those destroys `cos^2+sin^2=1` and "exactly one is 1".
Those got analytic divisor fixes in obs v2 instead.

Three things verified rather than assumed:

- **Safe with HER.** `compute_reward` and `_her_arrived` run on RAW arrays at
  `her_buffer.py:130-135`, *before* `_normalize_obs` at 136-137. So
  `arrival_eps=0.4` keeps meaning 0.4cm and there is no split unit system --
  which is the whole reason this route beat scaling the keys analytically (that
  would have touched nine call sites, including `eval_contact`'s distance
  binning).
- **The declared Box is untouched.** `VecNormalize.__init__` only rewrites
  `observation_space` for IMAGE spaces, so `check_for_correct_spaces` still
  passes and archived checkpoints still load.
- **The stats travel with the checkpoint, and eval REFUSES without them.**
  `vecnormalize.pkl` beside `model.zip`, `vecnormalize_best.pkl` beside
  `model_best.zip` (the stats drift, so best needs the stats as of the step it
  was best at). `eval_contact._load_vecnorm` raises in BOTH directions --
  flag-on-stats-missing and stats-present-flag-off -- because either way the
  policy is scored on an input distribution it never trained on, and that looks
  exactly like a bad checkpoint. All four cases exercised by hand.

The periodic-eval callback needed it too: its envs are BARE gym envs, so it now
takes an `obs_normalizer` and applies it **at the predict() call only** -- its
distance binning has to stay in raw cm or the bin edges move silently.

### P2/P3 -- the Gamma arrival dispatch, and a tolerance that travels

**P2.** `step` scored recontact-Gamma arrival with
`recontact_arrival(target=self._goal_xy[:2], direction=active_finger)`. Under a
Gamma goal `_goal_xy[:2]` is finger **L's** slot, so with R active it measured R
against L's target. Active is R about half the time -- matching the measured
254/500 (50.8%) of resets where a state perfectly achieving the intended 6-D goal
scored arrived. The correct 6-D conjunction existed but was reachable only from
`_her_arrived`, so **the rollout reward and the relabeled reward optimized two
different objectives**, and ~63 GPU-hours of Gamma arms are uninterpretable.

Now `step` routes through a new `_gamma_arrival`, which wraps `_gamma_arrived`
into an `Arrival`. `dist_to_target` is the WORST per-finger distance, matching a
new `_gamma_dist` -- `goal_dist` would have read finger L's slots only, i.e. the
same slicing mistake one level down. `_goal_dist_vec` does the batched version
for `compute_reward`.

**P3.** `_gamma_arrived` fell back to `self._gamma_tol`, and SB3 calls
`compute_reward` through `env_method(..., indices=[0])` -- so a whole relabeled
batch was graded against whatever interface **env 0** happened to be in at sample
time, not the one its own episode drew. The tolerance now rides per transition in
`info["gamma_tol"]`, the same pattern as `pre_achieved_goal` / `obj_settled`
(recontact already sets `copy_info_dict=True`).

**And the gate that should have existed.** `gamma_goal` was instantiated **ZERO
times** in a 159-check harness -- that is how this ran broken. Eight new checks,
the load-bearing one being: a state placed exactly on both fingertip targets with
both touch flags satisfied must score arrived **for both L-active and R-active**.
Measured 12/12 and 12/12; under the old code the R-active column was the bug.

### P4 -- a reward vector that can express contact, face, and stability

None of the three things you asked to encourage was expressible. `w_m` was ONE
scalar charged for ANY guard outcome (so contact-loss and wrong-face could not be
weighted apart), there was no per-tick contact term at all, and settling lived
inside a binary arrival flag. Added, all defaulting to inert:

- **`w_guard`** -- per-outcome penalties, keys from the new `GUARD_OUTCOMES`,
  falling back to `w_m` so an outcome added to a template's guard cannot silently
  become free.
- **`w_hold`** -- per tick, paid only on the conjunction "required contact held
  AND on the commanded face". One term, not two: separately they double-count.
- **`w_settle`** -- per tick, paid only while settled AND within
  `settle_radius_cm`. The proximity gate is load-bearing.
- **`w_prog`** -- POTENTIAL-BASED distance shaping. Gated by a round-trip test:
  a closed loop back to the start sums to 0.0, which absolute `w_d` does not
  (same loop costs -9.0). `w_d` charges the absolute distance every tick, so at
  0.005 a 20cm goal costs -20 over 200 ticks, TWICE the arrival bonus -- that
  arithmetic is the whole "bailing out was cheaper than attempting" story.
- **`w_arrive_pos`** -- one-shot bonus for position-only arrival, so the settle
  requirement is a gradient rather than a cliff (53% of push's failures already
  land within 1cm).

**`target_clip` and dense shaping are now REFUSED together.** The `[0, clip]`
bound assumes `Q* <= goal_reward`; with any per-tick penalty the true Q on a
failing state is negative, so the lower clamp biases the critic upward exactly
where it must learn "this is bad". `train_contact` raises. Worth stating plainly
that this is a real trade: `target_clip` is what took recontact 1/6 -> 6/6 seeds.

### TWO SELF-INFLICTED CHEATS, BOTH CAUGHT BY SMOKE RUNS RATHER THAN BY READING

Recorded because the pattern is the point: every dense term needs a metered cap,
and reasoning about the weights was not enough.

**(a) An uncapped per-tick bonus scales with the horizon.** At `w_hold=0.02` on a
200-tick horizon, a 2,000-step run reached **ep_rew_mean 4.57 with success_rate
0.0 and every episode at the full horizon** -- "hold contact and stall" was
already worth 46% of the arrival bonus. Same shape as v16's wall-parking cheat.
Fixed with `hold_cap` (default 2.0), mirroring `settle_cap`.

**(b) `w_arrive_pos` was documented one-shot and implemented PER TICK.** Position
arrival does **not** terminate the episode -- only `reached_interface` does, and
push has no overshoot guard -- so a policy parking at the goal while jittering
would have earned 3.0 EVERY tick: **600 against a 10.0 arrival bonus**. Now
credit-metered to fire once.

Measured effect of metering, same config and seed, 2,000 steps:

```
                      ep_rew_mean   success   ep_len
per-tick, uncapped           4.57      0.000    200.0
capped hold only             3.86      0.125    178.4
all three metered            1.62      0.125    178.4

max earnings WITHOUT arriving, 200-tick horizon:
  before   hold 4.00 + settle 0.50 + arrive_pos 3.00x200 = 600   -> 60x the bonus
  after    hold 2.00 + settle 0.50 + arrive_pos 3.00 once = 5.50, less 2.00 time
```

**Revised starting weights**, from this: `w_hold=0.005-0.01` (not 0.02),
`hold_cap=2.0`, `w_settle=0.02`/`settle_cap=0.5`, `w_arrive_pos=3.0`, `w_T=0.01`,
`w_prog=0` or small, `w_d=0` always, guards
`{contact_lost: 2, wrong_face: 2, forbidden_contact: 3, off_board: 5}`.

### A DIGEST TRAP, and the discipline that fixes it

Adding fields to `RewardWeights` moved the env digest of **every config in the
repo** -- `249434216cd2 -> 436dee0952c5` -- because the digest is a sha1 over
`repr()` of each env kwarg and a stock dataclass repr lists every field. Caught by
replaying v33 `ctl_s1`: all 60 episodes came back **bit-identical** while the
digest string had changed. That would have orphaned `logs/eval/v32_floor`,
`logs/eval/v32_final` and all of `logs/eval/v33`.

A *minimal* repr did not fix it either (`-> af4bf51bbb02`): the archived digests
were computed from the FULL seven-field string, so that exact string is what has
to be reproduced. `RewardWeights.__repr__` now emits the seven legacy fields
always, in their frozen order, then any NEW field only when it is set. Same
discipline as folding `adjacent` into the existing `guard_face` key in v33.

**Verified restored: `249434216cd2` and `a78252c0a0a6`, both with 0 of 60
episodes differing.** Gated by an equality check against the archived repr string
plus a check that a SET weight still moves it -- otherwise the trick would hide
real reward changes, which is worse than the bug.

`normalize_goal_keys` joined the interface-key list (now ten, cross-checked
across four files by `static`), so it too is outside the digest and obs-v1-vs-v2
and normalized-vs-raw are all arms scorable on ONE benchmark.

### Verification summary

- Gates **30 / 27 / 192 / 172 / 18**.
- v33 `ctl_s1` -> 0.683 / 0.917 / 7.42cm / 0.40cm / 67 ticks, digest
  `249434216cd2`, **0 of 60 episodes differing**. `recon_base_s1` -> 0.983,
  digest `a78252c0a0a6`, likewise.
- `normalize_goal_keys=true` trained, stats saved, scored; the refusal fires in
  both directions and `model_best` picks up `vecnormalize_best.pkl`.
- recontact-Gamma smoke-trained through the NEW dispatch with
  `init_gamma_modes=[free,push,pivot,pinch]`, pure sparse.
- dense push smoke-trained at the recommended weights with `target_clip=null`,
  and refused with `target_clip` set.

**Next.** P5 PPO, P6 recontact launcher parity + `finalize.sh --template` + the
object-frame video overlay, P7 along-face spawn randomization (prerequisite for
the diversity sweep -- a face-centre contact produces exactly zero torque), P8
housekeeping, P9 floor re-measurement (needed only for P7 now, since neither obs
v2 nor these four moved the digest).

## P5: the PPO path (2026-09-02)

Memo sec 4's "PPO as an algorithm-independence replication", Table 4's settings.
**All five gates green: 30 / 27 / 204 / 172 / 18** (`contact` 192 -> 204). SAC
re-verified bit-identical: v33 `ctl_s1` and `recon_base_s1` both replay with 0 of
60 episodes differing and their digests intact.

### The key is `rl_algo`, not `algo`

`algo=ppo` on the CLI dies with **"Could not override 'algo'. No match in the
defaults list."** `config/algo/{sac,ppo}.yaml` exists for nav, so Hydra parses
`algo=` as a config-GROUP override, and `config/train_contact.yaml`'s defaults
list has no `algo` entry. Renamed to `rl_algo`, with a gate asserting both that
the key is `rl_algo` and that `config/algo/` really exists -- otherwise a later
tidy-up renames it back and the collision returns.

### What PPO refuses, what it drops, and why the line is there

- **REFUSED: `use_her`, `target_clip`.** Both change what is being trained, so a
  launcher setting them for PPO means someone believes they are getting HER or a
  clamped critic. `use_her` has no replay buffer to relabel into (the memo notes
  HER "is not standard" for PPO and suggests goal-resampled on-policy rollouts,
  which is a separate build); `target_clip` has no TD target to clamp.
- **ANNOUNCED AND DROPPED: `learning_starts`.** It is an SAC-only knob that
  rides along in the SHARED protocol pins -- `logs/sweep_*/PINS.txt` carries
  `learning_starts=10000` -- so refusing it would force every PPO arm to edit
  the pin set, breaking the "PINS.txt is authoritative" discipline v33
  established after the hardcoded-portal bug. Printed, never silent.

  This distinction was found by running it: with the v33 pins, the refusal fired
  on every PPO invocation and made the flag unusable in a sweep.
- **WARNED, not refused: pure sparse.** PPO has no HER, so a one-shot arrival
  bonus gives it almost no gradient (push's untrained floor is 0.042 on goals
  >=3cm). A sparse PPO arm is a legitimate negative control, but it must be a
  deliberate one, so `train_contact` says so loudly and continues.

### Two conventions carried over rather than reinvented

**`n_steps` is the TOTAL rollout across envs**, divided by `n_envs`, exactly as
`train.py` and `config/algo/ppo.yaml` do it. SB3's own `n_steps` is PER env, so
getting this backwards changes the update size by a factor of `n_envs` with no
error. `n_envs` must divide it. Default 4096, inside Table 4's 2048-8192.

**`net_arch=null` resolves to `[256, 256]` for BOTH algos.** SB3's PPO default is
`[64, 64]` against SAC's `[256, 256]` -- about **16x fewer parameters** -- and
memo sec 9 requires the advantage to survive a control for total network
capacity, so shipping SB3's asymmetric defaults would have confounded the entire
comparison before it ran. Table 4's literal 3x256 is available as
`net_arch=[256,256,256]` and is a recorded deviation. **SAC is passed
`policy_kwargs` only when `net_arch` is set explicitly**, so its default path
stays byte-for-byte SB3's rather than depending on `[256, 256]` happening to
equal SB3's SAC default today.

### eval_contact branches on the value head, and labels the column

PPO has no critic. `hasattr(model.policy, "critic")` selects between SAC's
min-over-twin-Q(s0, pi(s0)) and PPO's `predict_values(s0)`. They go in the same
column because the comparison of interest is the same -- predicted value against
realized return -- but **they are different quantities**: V is the value of the
policy's own action distribution, not of the greedy action. The printout says
`V(s0) ... [V, not Q -- not comparable to a SAC gap]` so the two are never read
as one number.

`rl_algo` also joined the interface-key list (now **eleven**, cross-checked
across four files by `static`), so it is forwarded per cell by
`tools/score_sweep.py` from `meta.txt` and stays outside the env digest -- a PPO
arm and a SAC arm score on ONE benchmark.

**And a wrong `rl_algo` now raises a named error.** Loading a PPO zip through SAC
surfaced as `AssertionError: N-step returns are not supported for Dict
observation spaces yet` from deep inside the replay buffer, which reads like a
corrupt checkpoint. Wrapped: the message now names the checkpoint, the flag, and
where the flag comes from.

### Verification

- Every refusal exercised by hand: `use_her`, `target_clip`, a bad `rl_algo`
  value, `n_envs` not dividing `n_steps`, and the two non-fatal notices.
- **PPO trains.** 8,192 steps, `n_envs=4`, `n_steps=1024`, dense reward,
  `lr_linear_decay=true`: `explained_variance` climbed **0.009 -> 0.491**, so the
  value head is learning. `clip_fraction` 0.12 -> 0.08.
- **`net_arch` verified on the saved policy**, not just in the call:
  `[(256, 48), (256, 256)]`, and input 48 = obs 40 + achieved 4 + desired 4.
- **PPO checkpoint scored** at digest `249434216cd2` -- the same benchmark the
  SAC arms use -- reporting V(s0) with its caveat. 0.100 at 8k steps, which is
  an untrained number and quoted only to show the path runs.
- **PPO + `normalize_goal_keys` together**: trains, saves both stats files,
  scores.
- SAC bit-identical, both templates, digests intact.

**Next.** P6 (recontact launcher parity, `finalize.sh --template`, the
object-frame video overlay), P7 (along-face spawn -- a face-centre contact
produces exactly zero torque, so it gates the diversity sweep), P8 housekeeping,
P9 floor re-measurement for P7 only.

## P6-P9: recontact automation, the along-face spawn, the nested deletion, and a new floor (2026-09-03)

**All five gates green: 30 / 27 / 225 / 172 / 18** (`contact` 204 -> 225).
Archived checkpoints re-verified bit-identical at the end: v33 `ctl_s1` replays
to 0.683 / 0.917 / 7.42cm / 0.40cm / 67 ticks at digest `249434216cd2` with
**0 of 60 episodes differing on any recorded field**, and `recon_base_s1` to
0.983 at `a78252c0a0a6`.

### P6 -- recontact can now be swept and rendered like push

**`slurm/submit_sweep_recontact.sh` was a stale 67-line script** with none of
the v33 apparatus: no `PINS.txt`, no `GIT_DIFF_SHA`, no `uncommitted.diff`, and
the DEAD `squeue -o "%A_%a"` last-task test (on this Slurm `%a` renders as the
ACCOUNT name, so the block never ran on any sweep). Rewritten as Sweep D with
the full apparatus, plus two groups selected by `--export=ALL,GROUP=`:

- `GROUP=base` (3 cells): does obs v2 COST recontact anything? Changes obs and
  nothing else -- it pins `angular_drag_arm_cm=6.0`, the value `recon_base`
  actually trained on, and keeps `recontact.yaml`'s shaping. A replication that
  silently inherited the new 3.12 default would not be a replication.
- `GROUP=gamma` (9 cells): `gamma_free` / `gamma_init` / `gamma_init_shaped`.

**Two groups, two sbatch calls, two PINS files, deliberately.** The 2-D
single-finger goal and Eq 13's 6-D interface goal are different GOAL SPACES, so
`check_for_correct_spaces` refuses to load one against the other's env -- they
can never share a benchmark, and one PINS.txt would be wrong for three cells.

`init_gamma_modes` is a TASK key, so PINS pins the COMMON protocol at
`init_gamma_modes=[free]` (the harder, canonical acquisition task) and the
starting-in-an-interface arms need v33's two-way treatment. One bug caught while
writing it: `w_guard` makes `RewardWeights.dense()` true, so passing it to the
pure-sparse arms would have tripped the `target_clip` refusal at startup. It is
now scoped to the shaped arm through a bash ARRAY, which expands to nothing when
empty rather than to an empty argument Hydra rejects.

**`finalize.sh` recovers the template from the runs**, not from an argument:
`sed -n 's/^TEMPLATE=...' <sweep>/*/meta.txt | head -1`. Verified against both
real sweeps -- `43679344` -> push, `42617867` -> recontact. A hand-typed template
is one more thing that can disagree with what trained, which is the same reason
`PINS.txt` is read rather than retyped.

### P6 -- and the recontact video was broken TWO ways, both now fixed

**(a) The goal marker was drawn in the wrong place.** `Snapshot.goal_xy` is
documented as "the DESIRED OBJECT POSITION, not a finger target", and recontact's
goal is a FINGERTIP target in the OBJECT's frame. It was being forced into
`goal_xy` anyway. Added `finger_goals` / `finger_goal_tol_cm` (world frame, one
marker per fingertip with its own tolerance ring, since Eq 13's tolerances are
deliberately asymmetric at 0.3cm anchor / 2.0cm retracted).
`physics.to_snapshot` does the object->world transform PER FRAME -- not once per
episode -- because the object drifts and recontact's whole premise is that it
should not; a fixed world position would hide exactly that. `visualize.py` stays
frame-agnostic, per its own contract.

Two follow-ons the fix exposed: `goal_dist` now returns the WORST fingertip
distance for a two-finger goal (it was reading nan, so the caption and the
closest-approach marker were both dead for recontact), and `save_video` trails
the ACTIVE FINGER when the goal is a fingertip target -- trailing the object
drew a still dot and hid the only motion in the clip.

**(b) EVERY recontact mp4 was 0 bytes, with ffmpeg silent on stderr.** libx264
with yuv420p rejects an odd axis, and `save_video` passes
`macro_block_size=None` so nothing pads frames. The 80x60 recontact board
renders **509px** tall -- `6.0 * 60/80 + 0.6 = 5.1in`, and `5.1 * 100dpi`
truncates to 509 -- while push's 50x30 board gives 420. **That is why this only
ever broke recontact, and why the five archived v23 clips are the only recontact
media that exists.** Fixed by trimming the frame ARRAY (`_even_frame`), not by
choosing an even figsize: asking matplotlib for exactly 5.10in still renders 509.

Verified end to end: a `recon_base_s1` clip now writes 25-40KB mp4s, and the
rendered still puts the X target and its tolerance ring exactly where finger L
lands, with the closest-approach marker on the finger rather than the object.

### P7 -- the along-face spawn (`push_spawn_along_frac`)

The active finger spawned at the exact face CENTRE (measured max along-face
offset **0.0000cm** over 400 resets) against the memo's "random point along the
face". Three reasons that is not cosmetic, and the third is the load-bearing one:

1. it is a spec deviation;
2. the centre of a 10x6 object's long face is 3.0cm from the corner, ~4 policy
   ticks at `v_max`, so every episode started the same few ticks from a face
   change -- and policies leave the contacted face on 82-87% of episodes;
3. **a contact at the face centre pushing along the inward normal produces
   EXACTLY ZERO torque**, because the lever arm is parallel to the force. All of
   push's rotation authority came from the friction-limited tangential
   component, which is why net rotation measures a median 1.8deg/episode against
   a +/-45deg goal window. So this GATES the orientation-diversity experiment:
   widening the window without it only manufactures unwinnable episodes.

`None` draws no extra random number (bit-identical, verified 0/300 nonzero);
`0.7` gives 300/300 off-centre, max 3.475cm = 0.7 x the 5.0cm half-face. Capped
at 0.9 because a contact ON the corner makes `nearest_face` flip on rounding --
the same reason `contact_templates` caps its own interface sampler at
`ALONG_MAX_FACE=0.7`. The draw sits INSIDE the reverse sampler's 256-try
rejection loop; outside it, all 256 retries would reuse one offset and bias the
accepted set.

### P7 -- AND A DIGEST TRAP, the third of this kind

A new env kwarg rehashes every config and orphans every stored score. It fired:
the v32/v33 protocol moved **`249434216cd2` -> `e35ceab30ae5`** while v33
`ctl_s1` replayed bit-identically. Fixed with `stamp_omit_if_default` in
`eval_contact` -- a post-hoc TASK key is omitted from the stamp WHILE AT ITS
DEFAULT and rehashes only once actually set. Same discipline as folding
`adjacent` into the existing `guard_face` key, and as `RewardWeights.__repr__`.
Verified: `249434216cd2` with the key off, `646ba4ae1fd4` with it at 0.7.

**A float32 regression I introduced in P6 was caught by the same check.**
Routing `d0`/`min_dist` through the new arity-aware `_goal_dist` dropped the
float64 cast, moving `min_dist` in the 8th decimal on **57 of 60** episodes
while every other field stayed identical. Restored, and the comment says why the
`dtype=float` is load-bearing.

### P8 -- housekeeping, in the same digest-moving commit

**`angular_drag_arm_cm` default 6.0 -> 3.12.** `tau = mu*m*g*L` and for a body
sliding on a plane `L` is the pressure-weighted mean radius of the contact
patch: uniform pressure over a 10x6cm rectangle gives 3.12cm, and 5.83cm (the
half-diagonal) is the hard ceiling. **6.0 is above that ceiling -- no pressure
distribution can produce it.** Costed at zero by v29's `physdamp` arm (0.739 ->
0.706 on identical episodes, paired mean -0.033, holding under both
checkpoints), and every v32/v33 cell already passed 3.12 explicitly, so their
digests are unaffected.

**The nested curriculum is deleted.** Eq 15's literal nested form measured
INERT on this board -- same-room median 2.02/1.94/2.15/1.78cm across four levels
against 2.00 with no curriculum, because a nested level can only DELETE far
starts, never make near ones commoner. Checked before deleting rather than
after: **30 of 30 archived cells record `curriculum_mode=band` and 0 set
`curriculum_start_cm`.** Gone: `_range_cap`, `_start_window`, and
`_sample_room_xy`'s `x_window`.

**But the KEYS survive, because `curriculum_mode=band` appears in 2 archived
PINS.txt files** and `config/loader.py` rejects unregistered keys by design --
deleting the key would make those protocols un-replayable. So `nested` now names
only "the historical coned forward sampler, no ramp", which is what every
pre-v32 run actually did, and the combination that used to be silently inert
(`nested` + `curriculum_levels`) is the one that now raises. `curriculum_start_cm`
likewise raises if set. **The inert path is unreachable; the replayable
interface is intact.**

### P9 -- the floor, regenerated, and the success bar survives

`tools/make_v34_floor.sh` -> `logs/eval/v34_floor/`, 4 cells, one directory each
so its `vecnormalize.pkl` is unambiguous. `make_untrained_ckpt.py` now also
produces matching normalizer statistics when `normalize_goal_keys` is on, by
running the untrained policy for 2,000 steps and letting VecNormalize observe
the distribution it actually acts on -- otherwise `eval_contact` refuses the
checkpoint, correctly, since a policy scored on an input distribution it never
saw is not a floor for anything.

| cell | digest | all | **>=3cm** |
|---|---|---|---|
| `a1_v1_centre` (control) | `249434216cd2` | 0.067 | **0.042** |
| `a2_v1_along` | `646ba4ae1fd4` | 0.083 | **0.042** |
| `a3_v2_along` | `646ba4ae1fd4` | 0.067 | **0.042** |
| `a3_v2_along_raw` | `646ba4ae1fd4` | 0.017 | **0.000** |

**The control reproduces `logs/eval/v32_floor/untrained_pose_contact_frame`
exactly** -- same digest, same 0.067 / 0.042, same five bins -- so the archived
anchor survived every change between v33 and Sweep A and an A2-A1 difference is
attributable to the spawn.

**THE FLOOR DOES NOT MOVE WITH THE SPAWN: 0.042 on all three `contact_frame`
cells.** So THE PUSH SUCCESS BAR IS UNCHANGED and stays un-relitigated -- Bar 1's
>=0.40 is still ~10x the floor. **Raw actions floor at 0.000 on goals >=3cm**,
which is what Sweep C's `ppo_raw` arm is measured against.

One ZERO-SHOT preliminary, not a prediction: v33 `ctl_s1`, trained at the face
centre, scores **0.583** on the randomized-spawn protocol against 0.683 on its
own -- the spawn change costs about **0.100** to a policy that never saw it.
Sweep A's A2 arm turns that into a trained number.

### And the Sweep A launcher, so the claim above is actually true

`status.md` now says "nothing infrastructural blocks Sweep A", which was only
true with a launcher. `slurm/submit_sweep.sh` rewritten from v33's
scaffold-removal design to Sweep A: 9 cells = 3 arms x 3 seeds at **1.2M** steps
with `ckpt_freq=600000`, so the 600k snapshot supplies the budget axis for free
rather than costing three more arms. Wall time 8h -> 14h, since 600k took
4.5-5.1h. v33's launcher is preserved per-run at
`logs/sweep_43679344/*/submit_script.sh`.

**The common PINS protocol is the ALONG-FACE one** (A2/A3's task), not A1's: two
of three arms train there and it is what Sweeps B/C/D inherit. So A1 gets v33's
two-way treatment -- scored on the common protocol AND on its own
`249434216cd2`, which is what keeps it comparable to every archived v32/v33
number.

**Verified rather than assumed:** the PINS the launcher writes hash to
**`646ba4ae1fd4`**, matching `logs/eval/v34_floor/a3_v2_along` exactly, and the
floor cell re-scores identically through them. A new `static` check asserts each
launcher's PINS carries the TASK keys its own sweep moves -- `static` 30 -> 36.

**Next.** Commit, then Sweep A. Phase 0 is complete: P1-P9 all landed, and both
sweep launchers are ready.

---

## 2026-09-03 — pre-sweep audit: Bar 2 is already met, and four unfaithful-code fixes

**Question.** Before submitting Sweeps A and D, is the Phase-0 implementation faithful to
what it claims — and do the two cheap checks change the plan?

### Bar 2 is CLEARED, zero-shot, with no training at all

`tools/bar2_zeroshot.sh`, `logs/eval/v34_bar2/` (+ `PROTOCOL.md`). v33's frozen `ctl`
checkpoints re-scored under Eq 13's settled arrival. Valid on a frozen checkpoint because
`require_settled` changes the arrival TEST and nothing the policy reads, and does not touch
the reset sampler — so both digests draw the SAME 60 initial states and the comparison is
per-episode.

| seed | position-only >=3cm (`249434216cd2`) | settled >=3cm (`fdc2a41dc665`) | delta |
|---|---|---|---|
| s0 | 0.604 | 0.583 | -0.021 |
| s1 | 0.750 | 0.708 | -0.042 |
| s2 | 0.667 | 0.583 | -0.084 |
| **mean** | **0.674** | **0.625** | **-0.049** |

Settled untrained floor **0.000** on >=3cm (`floor_settled`, same digest). `model_best`
agrees: 0.646 / 0.583 / 0.688, mean 0.639.

**Bar 2 is >=0.40. Measured 0.625 against a 0.000 floor. It is met.** `docs/TODO.md` said
"expect this to be the expensive half" — **that prediction was wrong, and the price is
0.049.** Item 7 of ORDER OF WORK is done, and Phase B is unblocked without a training arm.

The position-only column reproduces v33's 0.674 **exactly**, which doubles as the
regression check on everything this session changed.

### |dtheta| in the eval report, and it says push does not rotate

`eval_contact.orientation_report`. Distance carries no gradient above 3cm on ctl_s1
(0.833/0.833/0.833 across the 3-6/6-9/9-12 bins); the orientation gap does. Measured on
ctl_s1:

- **42 of 60 benchmark goals (70%) are already inside the 22.5deg tolerance at reset.**
  Success on those 0.762, on the 18 that must rotate **0.500**.
- Mean |dtheta| **increases** by +1.36deg over an episode. On the untrained floor it
  increases by +14.02deg, and across the three ctl seeds it is +2.40 / +0.97 / -2.09.

So push's pooled success is substantially a feasibility artifact, and the policy has
close to no rotation authority — which is the face-CENTRE spawn's zero-torque geometry,
measured rather than derived. This is why `contact_descriptors` must be built on
orientation, and why Sweep A prices the spawn as its own arm.

### The Gamma floor now exists

`tools/make_v34_recontact_floor.sh`, `logs/eval/v34_recontact_floor/`. Sweep D's header
promised a floor that was not on disk. Both cells' digests were verified by re-scoring
them through the launcher's own `PINS_LINE` text.

| group | digest | floor |
|---|---|---|
| `base_v2` | `1ecc01e69a3d` | **0.033** all bins (2/60) vs `recon_base`'s 0.978 |
| `gamma_free` | `5dff6e0afd4a` | **0.000**, 0 of 48 |

Two facts that change how a gamma number reads: the stratified sampler yields **48
episodes, not 60** (worst-fingertip distance has a 15.8cm median at reset, so the 0-3cm
bin is unfillable), and `object_disturbed` fires on **22.9% of untrained episodes**.

### Four faithfulness bugs, all found by running rather than reading

1. **`w_arrive_pos` paid PER TICK on the HER path.** `step` meters it once per episode;
   `compute_reward` had no episode history and paid 3.0 on every position-arrived
   relabeled row — measured, i.e. 3.0/(1-0.99) = **300 of implied Q against
   goal_reward=10**, on ~80% of every batch. Exactly the rollout/relabel split the Gamma
   arrival bug was. Fixed by refusing `w_arrive_pos` with `use_her` and deleting the term
   from `compute_reward`; Sweep D's shaped arm now uses **`w_prog=0.1`**, which is
   potential-based and relabels exactly (Ng et al. 1999).
2. **`base_v2` could not start.** The blanket "target_clip is refused with any dense
   reward" rejected recontact's OWN archived baseline (`w_T/w_a/w_m` + `target_clip=10`,
   what `recon_base` scored 0.978 with). WHICH END of `[0, clip]` a term breaks depends on
   its SIGN: negative-only shaping keeps `Q* <= goal_reward` so the upper clamp is sound,
   while positive shaping (`w_hold`+`w_settle`+`w_prog`) exceeds 10 and the clamp deletes
   the value the shaping exists to create. Split into `positive_shaping()` (refused) and
   negative-only (announced, kept replicable).
3. **`w_prog` was silently absent from ~80% of every batch** unless `min_progress_*` or
   recontact happened to force `copy_info_dict`. It reads `info["pre_achieved_goal"]`.
4. **`tools/make_untrained_ckpt.py` hardcoded `_make_env("push", ...)`**, so
   `contact=recontact` produced a PUSH floor — wrong template, goal space and horizon.
   Found while generating the Gamma floor.

Also: a w_guard key outside `GUARD_OUTCOMES` now raises instead of falling back to
`w_m=0` and being silently free; `reward.RELABEL_DROPPED` records what HER cannot
reconstruct and each term's per-episode bound (w_T*horizon 2.0, hold_cap 2.0, settle_cap
0.5 — bounded on purpose, unlike w_arrive_pos); dead field `w_face` deleted.

### Sweep D gained a fourth arm

`gamma_init_shaped` moved shaping AND `target_clip` at once, and the clamp is what took
recontact from 1/6 to 6/6 seeds — so a failure would have been uninterpretable in exactly
v33's way. **`gamma_init_noclip`** (gamma_init, `target_clip=null`, nothing else) makes it
a bisection: `gamma_init - gamma_init_noclip` prices the clamp,
`gamma_init_shaped - gamma_init_noclip` prices the shaping. 9 -> 12 cells.

### Verified

Gates **36 / 27 / 243 / 172 / 18** (contact 225 -> 243). v33 `ctl_s1` replays with **0 of 60
episodes differing on any of 11 fields** at digest `249434216cd2`, success 0.6833... exactly.
All 15 Sweep D cells and all 3 Sweep A arms start and train from the real launcher text;
the PPO path runs; both new refusals fire when they should.

**Next.** Submit A and D. Sweep B's rotation arm should be read against the 70%/0.500
split above, not against the pooled number.

---

## 2026-09-03 — v34 SUBMITTED: 24 cells across three jobs, and the first signal

**Question.** With Phase 0 complete and the audit's nine fixes landed, run the two sweeps
that Phase 0 existed to enable.

### What is running

| job | sweep | cells | budget |
|---|---|---|---|
| `44180162` | A, push | 9 (`a1_v1_centre`/`a2_v1_along`/`a3_v2_along` x 3 seeds) | 1.2M, `ckpt_freq=600000` |
| `44180252` | D-base, recontact | 3 (`base_v2` x 3) | 1M, horizon 100 |
| `44180185` | D-gamma, recontact | 12 (`gamma_free`/`gamma_init`/`gamma_init_noclip`/`gamma_init_shaped` x 3) | 1M, horizon 200, array `%4` |

All 24 cells: `GIT_COMMIT=f0bb3bd`, `GIT_DIRTY=yes`,
**`GIT_DIFF_SHA=509bcc5e0fbe23b2`** — identical across all three jobs, so every cell ran the
same code. The working tree was verified byte-identical to the archived `uncommitted.diff`
(zero diff-of-diffs lines), so the code is fully recoverable. **Still not committed.**

Outputs will land in `logs/eval/{sweepA,reconD_base,reconD_gamma}/`. Expected digests and
their floors are tabulated in `status.md` under HOW TO SCORE V34.

### The early signal — diag eval only, NOT cross-cell

Each cell's own training-time eval, 32 episodes, its own reset distribution. **A1 and A2/A3
sit on different digests, so these columns cannot be subtracted.** Recorded because the
direction is already informative, at ~600-670k of 1.2M for A:

| arm | last | best | slope /100k |
|---|---|---|---|
| `a1_v1_centre` | 0.531 / 0.594 / 0.625 | 0.625 / 0.625 / 0.688 | +0.032 / +0.011 / +0.076 |
| `a2_v1_along` | 0.688 / 0.750 / 0.719 | 0.781 / 0.875 / 0.812 | +0.047 / +0.065 / +0.038 |
| `a3_v2_along` | 0.500 / 0.688 / 0.625 | 0.688 / 0.781 / 0.719 | -0.021 / +0.011 / +0.021 |
| `base_v2` (at 1M) | 0.969 / 1.000 / 0.969 | 1.000 / — / 1.000 | +0.10 to +0.18 |
| `gamma_free` s0/s1/s2, `gamma_init_s0` | **0.000** | **0.000** | +0.000 |

**Result 1 — the along-face spawn looks like it HELPS, and that inverts the prediction.**
The zero-shot measurement had v33 `ctl_s1` at 0.583 on the randomized protocol against 0.683
on its own, which read as a harder task. Trained on it, A2 is the strongest arm on every
seed. This is what the zero-torque finding predicts once you train on it: a face-centre
contact produces exactly zero torque, so off-centre contact is the only thing that gives
push real torque authority, and the more varied task is also the more learnable one.
**Provisional — needs the shared benchmark.**

**Result 2 — Gamma is 0.000 at 1M on 4 of 4 finished cells, and this is the headline
negative.** `gamma_free` on all three seeds and `gamma_init_s0`, each over 200 diag evals:
zero successes, zero slope, from step 4k to step 999k. The arrival bug is fixed and verified
(a perfectly-achieving state now scores 500/500 where it used to score 254/500), the
untrained floor is 0.000, and the trained result is 0.000. **Fixing the bug did not rescue
Eq 13's canonical interface.** `gamma_init_shaped` is still queued and is the last untried
variant.

That is consistent with v31's diagnosis rather than a surprise: the 4-way conjunction
essentially never fires by chance — L inside its 0.3cm anchor tolerance in 1/60 episodes, R
inside 2.0cm in 2/60, both touch flags matching in 4/60 — and HER can only relabel toward
goals a trajectory actually achieved. **If `gamma_init_shaped` is also zero, the conclusion
is that the conjunction is not acquirable as posed, and the next move is task design, not
budget:** staged or sequential fingertip goals, looser per-finger tolerances, or the `pivot`
template. Re-running gamma at a larger budget on the strength of a flat zero curve would be
the v28 mistake in reverse.

**Result 3 — obs v2 is free for recontact.** `base_v2` sits at 0.969-1.000 on diag against
the archived 0.978 benchmark, with slope still positive at 1M, despite obs going 17 -> 38
dims. The replication question answers yes.

### Timing, measured from the CSVs

At 14:52 EDT: A all 9 at 49-56%, ~3.3-4.3h left against a 14h wall. D-base s0/s2 done, s1
at 64% (~2.4h; it runs at 42 steps/s against s0/s2's 93, so it is sharing a GPU). D-gamma
4 done, 4 running (~2.7h), 4 queued behind `%4` — so gamma lands last at ~6h.

**Next.** Commit; confirm `finalize.sh` fired (it never has); score all three against the
pinned protocols; read A1 @1.2M first, because if it lands materially above 0.674 every v33
scaffold price was measured at an unconverged budget.
