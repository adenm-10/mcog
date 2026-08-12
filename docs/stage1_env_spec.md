# Stage 1 environment spec — planar contact maze, first pass

Deliverable #1 of memo sec 11. Covers exactly the push + recontact study (memo
sec 3.1, sec 8 Stage 1). Pivot/pinch and multi-room boards are later passes;
nothing here should be read as final past this scope.

## Embodiment (locked, memo sec 11)

Two independently actuated planar fingertips (circular discs), not a
parallel-jaw gripper. Dynamic bodies with impedance/PD control toward a
commanded velocity — **not kinematic** — so the object can push back on a
finger. This is deliberate: a kinematic (infinite-mass) fingertip erases the
underactuation that makes push/pivot/pinch a real contact-mode problem.

## Units — read this before touching any number below

**Internal sim length unit is the centimeter, not the meter.** PyMunk's
default collision tolerances (`collision_slop`, `collision_bias`) are tuned
for bodies of size ~1 in whatever unit you pick; our fingertip radius
(0.012 m) is far below that at meter scale, which makes contact resolution
mushy. At cm scale, the fingertip radius (1.2), object size (6–10), and board
(60–80) all sit in a sane range relative to a `collision_slop` of a few
hundredths of a unit.

Mass is in kg, time is in seconds. **Consequence: forces in this sim are in
kg·cm/s², i.e. 1 sim-force-unit = 1e-2 N.** Anything compared against a
memo-stated force in Newtons (a future guard threshold, Eq 14's `w_F` term)
must be converted. **Record this at the next sim revision** — if Stage 4
moves to MuJoCo/SI meters, every force-scale constant in this doc needs
re-deriving, not just relabeling.

## State

17 floats, oracle (no observation noise, no state estimation — memo sec 8/11):

```
[0:2]   object position (x, y), cm
[2:4]   object heading (cos, sin)
[4:6]   object linear velocity (vx, vy), cm/s
[6]     object angular velocity, rad/s
[7:9]   left fingertip position, cm
[9:11]  left fingertip velocity, cm/s
[11:13] right fingertip position, cm
[13:15] right fingertip velocity, cm/s
[15]    left-finger/object contact flag, oracle boolean (0.0/1.0)
[16]    right-finger/object contact flag, oracle boolean (0.0/1.0)
```

Contact flags come from PyMunk collision `begin`/`separate` callbacks, not a
force threshold — PyMunk gives an exact touching/not-touching boolean for
free, so no `λ_min` needs to be guessed yet. A numeric force threshold is
deferred until (if) grazing contact turns out to pollute a guard — not needed
for push/recontact.

## Action

`a = (vLx, vLy, vRx, vRy)`, each component in `[-1, 1]`, scaled by `v_max`
(20 cm/s starting value) to a commanded fingertip velocity. Each fingertip is
a dynamic body with a velocity-servo force `F = k_v (v_cmd − v_current)`
applied every physics substep — this is the "impedance" inner loop; `k_v` is
an initial hyperparameter (see below), not a derived constant, and should be
retuned once you watch a real push.

## Geometry and dynamics (fixed, not randomized — locked this session)

| Quantity | Value | Note |
|---|---|---|
| Board | 80 × 60 cm | memo sec 3.1.2, converted to cm |
| Object | rectangle, 10 × 6 cm | rectangle first; T-shape is a later variant |
| Object mass | 0.20 kg | midpoint of memo's [0.15, 0.30] kg range, fixed since we're not randomizing yet |
| Object–table friction (μ) | 0.40 | midpoint of memo's [0.20, 0.60]; **modeled as a manual Coulomb drag on the object body**, not a PyMunk contact — see below |
| Fingertip–object friction (μ_f) | 0.75 | midpoint of memo's [0.5, 1.0]; native PyMunk `shape.friction` |
| Fingertip radius | 1.2 cm | memo's 0.012 m |
| Fingertip mass | 0.05 kg | light, actuated; not in the memo, chosen so the servo settles quickly (see gains below) |
| Wall thickness / friction | 0.3 cm / 0.30 | cosmetic, only matters if something grazes a wall |

**Why table friction can't be a PyMunk contact:** this is a top-down 2D
simulation — the table is the plane of motion, not a shape in it, so there's
no "floor" for PyMunk's contact solver to apply Coulomb friction against.
Object–table friction is instead applied by hand each substep: a drag force
`F = −μ·m·g_eff·v̂` opposing the object's linear velocity (and an analogous
damping torque for angular velocity), with `g_eff = 981 cm/s²` standing in for
the normal force gravity would provide in a real top-down setup. This is a
simplification (real friction isn't a clean point-Coulomb law under a
distributed contact patch, especially for the rotational term) — flagged
here, not hidden in a comment.

Friction/mass are **fixed values, not sampled ranges**, on purpose: Stage 1's
first pass mirrors Stage 0's single-seed, deterministic-dynamics approach.
Domain randomization is Stage 2's question (memo's own staging) — turning it
on now would conflate "does composition work at all" with "does it survive
domain shift," which makes any failure hard to attribute to either cause.

## Control timing

| Quantity | Value |
|---|---|
| Physics rate | 500 Hz (`dt_phys` = 0.002 s) |
| Policy rate | 25 Hz → 20 physics substeps per policy step |
| Option horizon `T_o` | **200 policy ticks**, for the 3-room/90×60 cm corridor board specifically (see "First multi-room board" below) — measured, not assumed. A scripted push through the full route took 33/62/20 ticks per edge (worst case 62); 200 gives >3x margin on the worst observed edge. **This is a property of this board's geometry and gains, not a universal constant** — a board with bigger rooms or different gains needs re-measuring, the same way nav's `h_region` was maze-specific. |
| Episode budget | **600** = `(hops+1) * T_o` for a 2-hop route, matching nav's own "`h+1` options run for `h` hops" convention. |

## Impedance / drag gains

| Quantity | Value | Basis |
|---|---|---|
| `k_v` (finger velocity-servo gain) | **10.0 kg/s** (smoke-tested; was 3.0) | The original 3.0 guess's max force (`gain*v_max`=60) was *below* the table-friction force the object must overcome (`mu*m*G_EFF`=78.5 kg·cm/s²) — pushes crawled at ~0.44 cm/s instead of tracking the 20 cm/s command. 10.0 clears that with ~2.5x margin (max force 200) while staying well inside the stability ceiling (`k_v*dt_phys/m_finger`=0.4, vs instability risk near ~2 for explicit substepping — the original "~0.2" caution in this doc was overly conservative). |
| `collision_threshold_cm` (-> PyMunk's `collision_slop`) | 0.05 cm | A few % of fingertip radius (1.2 cm); tightened from PyMunk's default 0.1 because precise contact discrimination is the point of this study. Smoke-tested: zero contact-flag flicker over a sustained push. |
| `collision_bias` | PyMunk default | Dimensionless (a per-step overlap-correction fraction), doesn't need rescaling for the cm-unit change. |
| `angular_drag_arm_cm` (effective lever arm for the manual rotational-friction torque) | **6.0 cm** (smoke-tested; was 1.0) | Found via the 3-room board work, not the original push smoke test — every push tested before then was perfectly centered (zero lateral offset), which never generates torque, so this was never exercised. At 1.0, a push offset by just 1 cm from the object's center spun it to **-81 degrees** before losing contact: the servo's torque authority vastly outweighed that little damping. At 6.0 (roughly the object's own scale), the same 1 cm offset stays at -6 degrees and completes cleanly. Raising it further (10, 20 cm) doesn't raise the tolerable-offset ceiling past ~1.5 cm — that ceiling looks like a real physical limit of single-point-contact pushing, not an undertuned constant. |

**Smoke test result (this session):** at `k_v`=10.0, a continuous push
accelerates to ~11.7 cm/s in the first 2s, decelerating only as the object
approaches the far wall (confirms the wall, not the servo, is what stops it);
contact is stable (0 flicker) from first touch onward; a fingertip
repositioning on a path that doesn't intersect the object produces exactly
zero object drift. The `k_v`=3.0 guess is what originally motivated this
recheck — see the smoke-test note above.

## Settling / arrival thresholds (locked this session)

| Symbol | Value | Used for |
|---|---|---|
| `ε_v` | 0.5 cm/s | linear-velocity settle threshold |
| `ε_ω` | 5 deg/s | angular-velocity settle threshold |
| `ε_c` | 0.1 cm | position/contact tolerance — this **is** `arrival_eps` in the shared `ExecConfig`, not a separate knob |
| `n_grace` | 5 steps (~0.2 s at 25 Hz) | contact-loss grace period for the guard (deferred, see below) |

`ε_v`/`ε_ω`/`n_grace` are domain-local module constants in
`domains/contact_templates.py` (mirroring `HEADING_CONE_ALPHA_DEG`'s existing
pattern) — they do not require any change to `ExecConfig`/`executor.py`.

## Templates in scope

**Push and recontact only.** Pivot and pinch are later (memo sec 7 step 4's
ordering).

- **Push**: target = object position. Loose arrival = object within `ε_c` of
  target. Strict arrival (the canonical-interface / handoff-safe version,
  sec 2.4) = loose arrival **and** the object is settled (`ε_v`, `ε_ω`).
- **Recontact**: target = a position for one named fingertip (`direction ∈
  {"L", "R"}`, reusing the existing `direction` slot in the shared
  `score_arrival` signature rather than adding a new hook parameter). Loose
  arrival = that fingertip within `ε_c` of its target. Strict arrival = loose
  arrival **and** that fingertip is settled **and** the object is settled
  (interpreting "object static" from memo sec 3.1.1 as low-velocity, not
  zero-displacement-since-option-start, since `score_arrival` only sees the
  current state, not the option's entry state).

**Training scope decision (locked this session):** recontact's training
substrate is fully independent of push's — it never crosses a portal, so it
needs no board/`Portal` config at all, just a single open region with its own
entry sampler (vary the moving finger's start position and its target contact
point around the object's perimeter). It *could* therefore be trained on its
own, separately from push, at any time. **For now, train it the same way as
every other policy** (same reward/gym-env/train.py machinery, run back-to-back
or alongside push) rather than standing up a separate simplified pipeline —
revisit only if recontact turns out to need different treatment once real
training data exists.

## Guards — implemented

Unlike nav's `guard_region` (which only counts violations), the contact
guard **terminates the option** (`domains/contact_templates.py`'s
`push_guard`/`recontact_guard`, wired through `executor.py`'s widened
`guard_ok` contract — see the "Guard-terminating change" note below) on:

1. a required contact lost for more than `n_grace` steps (push only —
   recontact has no required contact by design),
2. a forbidden contact appearing (push only: the passive finger touching),
3. the object leaving the board (universal),
4. force exceeding a safety threshold (universal; the threshold itself is
   still unset — `force_abort_kgcms2=None` means this check never fires, since
   no telemetry yet justifies a number, even though the peak-force channel
   that would feed it is now measured every tick).

Recontact deliberately skips 1 and 2 — no required contact, and "object
stays still" during the move is trusted rather than checked (no drift guard),
since this pass has no domain randomization to make the sim misbehave.

**Guard-terminating change to shared code:** `run_option` (`executor.py`)
now breaks and labels the option's outcome when `guard_ok` returns a `str`
(one of four new `OPTION_OUTCOMES`: `contact_lost`, `forbidden_contact`,
`off_board`, `force_limit`), rather than only counting a `False`. nav's own
`guard_region` still returns a plain bool, so its behavior is provably
unchanged — verified by re-running the frozen `tests/fixture_eval.py` (tol=0)
gate after this change.

## First multi-room board: a 3-room corridor

Purpose: prove the plumbing works (region membership, portal crossing,
chaining two push options back-to-back via `run_episode`) before spending
effort on anything with turns, pivots, or recontact bays — the manipulation
analogue of the simplest possible nav sanity check.

| Quantity | Value |
|---|---|
| Board | 90 × 60 cm (widened from the single-room 80×60 so 3 rooms land on round numbers) |
| Rooms | `[0,30]`, `[30,60]`, `[60,90]` in x, full 60 cm in y |
| Portals | x=30 and x=60, both y∈[25,35] (10 cm gap — object's 6 cm short side plus margin). **Same y-range in both** → the whole crossing is a straight line, needing only `push`, never a turn |
| Start | object at (15, 30) — center of room 0 |
| Goal | (70, 30) — inside room 2 |

Verified with a real chained episode (`run_episode` + `planner.bfs_route`,
both untouched generic code, driven by a scripted push-toward-target
policy): route `[0,1,2]` found correctly, all three legs (`0→1`, `1→2`,
`2→goal`) reported `reached`, zero guard violations, episode `success`.
This is the first time `run_episode`'s full route-chaining path has run in
the contact domain at all — `run_option` alone was tested before.

**Region abstraction is now real**, not a placeholder: `domains/contact/
board.py`'s `Board` derives `region_of`/`adjacency`/`resolve_target` from
`PlanarFingertipParams.portals`. With zero portals (the original single-room
default) this correctly degenerates to one region and `resolve_target`
raises for any non-terminal edge — the same behavior the old hardcoded stub
had, now one case of the general logic instead of a separate code path.

**`push_arrival` is now crossing-based for non-terminal edges**: reaching a
portal means the object's edge crossing the line within the gap (memo sec
2.1's "portal set"), not landing on an exact point. Terminal edges
(`iface=None`) keep the original point-distance test.

**One simplification specific to this corridor, not a general rule:**
`resolve_target` hardcodes `direction="L"` for every edge (only the left
fingertip ever pushes, since the corridor only goes rightward), and
`contact_hooks(default_finger="L")` covers the terminal leg too (`run_episode`
never calls `resolve_target` for the final push-to-goal leg, so it never gets
a `direction` assigned otherwise). A board that pushes in more than one
direction, or with both fingers, needs real per-edge logic here, not a
constant.

## Learned-policy training design (locked this session, not yet implemented)

Recorded before writing any training code because these are structural
choices that would be expensive to unwind later, not tunable constants.

**Reward — memo Eq 14, sparse arrival + shaped terms, evaluated every step:**

```
r_e(s,a,s') = R_g * 1[s' in G_e]
             − w_d * d_e(s')                      object-centric distance to target interface
             − w_a * ||a||^2                       action-effort penalty
             − w_F * [F(s') − F_max]_+^2            squared hinge — zero unless force EXCEEDS F_max
             − w_m * (1 − χ_k(e)(s', a))            guard indicator, {0,1}, evaluated every step
             − w_T                                 constant per-step time cost
```

Two details easy to get wrong from a casual read: the force term is a
**squared hinge past `F_max`**, not a symmetric distance (staying below
`F_max` costs nothing) — our peak-force telemetry (`IDX_PEAK_FORCE`, already
tracked every tick) feeds this directly. And the guard term `χ_k(s,a)` is
our existing `push_guard`/`recontact_guard` — it already runs every tick for
termination, so the reward term is a free reuse, not new guard logic; it
shapes reward continuously, before a violation is severe enough to actually
terminate the option.

`R_g, w_d, w_a, w_F, w_m, w_T` start as placeholder values (same status as
`k_v`'s original guess) — set to a small sane default, smoke-tested for
sane-magnitude reward traces before any real training run, then revised with
justification exactly like the impedance gains above. Not blocking
implementation.

**Goal representation — a pose is a narrow "target set," not a separate
mechanism (memo Eq 6's node `v=(C_i,Θ_j,Γ_ℓ)` is already position **and**
orientation **and** interface typed; Algorithm 1 line 4 samples `g_e ⊂
R_w(e)`, the whole successor region, not just its portal).** A goal is
`(target_pose, tolerance_shape)`:

- **Portal-crossing goal** — tolerance wide along the portal's free axis,
  tight along the crossing axis (our existing `Portal(x, y_lo, y_hi)` shape,
  already how non-terminal `push_arrival` works).
- **Specific-pose goal** — tolerance tight in x/y (`ε_c`) and, when the edge
  needs it, in orientation too; optionally requiring settling (`ε_v`/`ε_ω`,
  our existing loose/strict arrival split), for edges where handoff safety
  matters.

Training samples both shapes across curriculum from the same successor
region, per Algorithm 1 — this is what produces "reach any pose in a room"
and "cross this doorway" from one policy/one loop, not two.

**Why this specifically matters for the offset-door finding
(documented below):** a *settled, specific-pose* goal near a portal is
the mechanism the memo intends to fix exactly the handoff-carryover failure
found there — training pressure to arrive slow and centered, not just
"somewhere in the loose portal band." Worth re-testing the offset case with
a trained policy once this exists, per the earlier decision to defer it
rather than hand-script a fix.

**HER relabeling — a real, currently-unresolved gap, documented rather than
silently narrowed:** the memo's Algorithm 1 line 11 relabels using "achieved
object poses **and portal sets**" — i.e., a rollout that missed its sampled
goal but still crossed *some* portal should be relabeled as a success for
that portal-set, not only as a success for the exact pose it stopped at.
SB3's stock `HerReplayBuffer` only supports the first half (substituting an
achieved pose as a new tight-tolerance point-goal); it has no notion of
"this achieved trajectory also satisfies this other, wider set-goal." First
implementation will do point-pose relabeling only (a real subset of what the
memo specifies, not a redesign) and flag portal-set relabeling as a
follow-up needing a custom goal-selection strategy — analogous to how
`force_abort_kgcms2` is deferred until data justifies it, not treated as a
blocker for starting training.

## Tried and deliberately not solved: offset (zigzag) doors

Attempted a harder version of this board — door 1 low, door 2 high, forcing
a diagonal push instead of a straight line — to test whether the plumbing
holds up under a more interesting path. It doesn't, for reasons worth
recording precisely rather than re-discovering later.

**Finding 1 — `angular_drag_arm_cm` was a real bug, now fixed (1.0 → 6.0,
see the gains table above).** Every push tested before this board was
perfectly centered, so the object's rotational-friction damping was never
exercised. A push offset by just 1 cm spun the object to -81 degrees at the
old value. Fixed by raising the effective damping arm to roughly the
object's own scale. This fix is real and carries forward regardless of the
zigzag question.

**Finding 2 — even fixed, a single continuous single-finger push only
tolerates ~1.5 cm of lateral offset, and this appears to be a genuine
physical limit, not a remaining tuning gap:** pushing the damping arm well
past 6.0 (10, 20, even 60 cm) doesn't raise the tolerable offset further.

**Finding 3 — a second, distinct issue: handoff velocity.** Traced the
object's state at the exact tick `push_arrival`'s crossing test
(`reached_position`) fires for a non-terminal edge: it was moving at
**~11.5 cm/s and rotating (2.35°, 0.13 rad/s)** — `reached_interface`
(settled) was `False` at that instant, confirmed directly. Fed that exact
exit state into a follow-up edge with **no new steering demand at all**
(same target y, straight continuation) and it *still* lost contact 31 ticks
later. So the residual motion alone, independent of any new diagonal
command, is enough to break a chained push. This is memo sec 4.4's handoff
compatibility problem ("the predecessor's terminal distribution may lie
outside the successor's training distribution"), observed directly rather
than inferred: `reached_position` (loose, crossing-based — the right choice
for matching nav's doorway philosophy) does not imply anything about the
handoff being safe for what comes next.

**Decision:** did not chase this further with scripted heuristics — tried a
proportional/decelerating policy (not just bang-bang aim-at-target) and it
still failed, just later. Reliably chaining off-axis pushes looks like it
needs either active braking/re-stabilization near each handoff or a policy
that has actually learned to maintain contact under disturbance — exactly
the gap SAC+HER training exists to close, not something worth hand-scripting.
**The aligned (straight-line) board above is the one taken as the plumbing
correctness proof.** The offset case is an open, documented finding: a
concrete, useful thing to point a trained policy at later to see whether
learning succeeds where scripting couldn't, not a task to force through now.
