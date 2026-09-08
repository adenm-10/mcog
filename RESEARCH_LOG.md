# Research log

**Who this is for: a human, between the proposal and the numbers.** What each
experiment asked, what it answered, and where the memo asked for it.

Newest first. **No provenance lives here** — every row points at
`docs/PROGRESS.md`, which owns the digests, job ids and confidence intervals.
If a number appears below it is a headline, not a record.

The memo is `docs/hierarchical_contact_rich_manipulation_research_plan.pdf`.

---

## 1. Scoreboard

| # | when | the question | the answer | memo |
|---|---|---|---|---|
| **Sweep D** | *running* | Is the contact interface acquirable at all? | — | Eq 13, sec 6.1 |
| **Sweep C** | *running* | Which action space, Gamma encoding and observation version does everything downstream inherit? | — | Eq 7, Eq 18, Eq 40 |
| Smoke | 09-08 | Does board v2 train off the floor before we spend ~220 GPU-h? | Yes — the curriculum advances (level 0 → 1). An earlier 2-cell smoke was cancelled at 62% showing no movement. | — |
| **Sweep B** | 09-08 | Budget or treatment: which moves push more? | **Budget, by 3-6×.** Neither treatment survived. And **the arm ordering flipped between 1.8M and 2.4M** — so one budget can rank arms backwards. Every sweep since carries four rungs. | sec 5.2 |
| Sweep B | 09-08 | Is push still gradeable by distance? | **No.** The distance spread collapsed across rungs and at the largest budget success *rises* with distance. Distance is dead as a `p_hat` feature on this board; **orientation survives.** | Eq 22 |
| Sweep A | 09-04 | Does the spawn position or observation v2 help push? | Neither, on either checkpoint. Budget dominated both. | sec 4.2 |
| **Stage 0** | closed | Does knowing *where* a skill fails predict multi-step success better than assuming it always works? | **Yes, and it is the project's headline.** The handoff-aware predictor beat both naive alternatives on every measure, and the advantage *grows* with route length. Passed its preregistered bar. | Eq 22, 29, 31 |
| Stage 0 | closed | *Why* does it work? | Legs fail because they **run out of clock**, not because they are hard: the break falls exactly at the distance the step budget buys. Confirmed on thousands of legs. | sec 5 |

---

## 2. Decisions taken, and what each cost

Each was measured before it was adopted. Full reasoning: `docs/TODO.md`
("DECISIONS TAKEN"). **These are not re-litigated.**

| | decision | why | the price |
|---|---|---|---|
| **D1** | Gamma is `{free, one-contact, two-contact}` — the commanded contact *face* is retired | a face index is meaningless on the memo's own next object (a T-shape), on a round object, in 3D, or on a soft body. A contact *count* is topological and survives all four | pinch ↔ pivot becomes invisible — a real transition the abstraction cannot express. Recorded, not hidden |
| **D2** | The "don't move the object" rule becomes a **displacement** bound, not a velocity one | the invariant exists so the *next* skill finds the object where it expected it. That is a displacement property. The velocity reading forbade the unavoidable contact transient and switched the learning signal off for ~96% of an episode | a recorded deviation from the memo, with its measurement |
| **D3** | Push cone 30° → 75° | measured from the reachable set, not guessed. ±180° is not supportable | none; it was measured rather than swept |
| **D4** | Board v2 is the benchmark: bigger, three rooms, **offset** doorways, goals in the far room | v1's doorways were aligned, so "cross the room" never required steering; and the goal sat *in* the doorway, so crossing only meant reaching the wall | a new digest — every earlier score is on the old board |
| **D5** | The face guard stays **off** during the interface sweep, and is priced zero-shot afterwards | comparing two floored arms answers nothing | the guard's cost is a delta, never pooled with the headline |

---

## 3. Memo coverage — asked vs answered

| memo asks for | status |
|---|---|
| Eq 22 `p_hat`, a learned edge-success model | **done for the maze.** Not yet fitted for contact — the ladder is unbuilt |
| Eq 29 `H`, handoff-aware composition | **done, and it is the Stage 0 result** |
| Eq 31/32 route cost | **done** (maze). Risk-aware routing implemented but **never run** |
| Eq 35 total interaction budget | tracked per round |
| Eq 13 target sets / interfaces | contact side redefined by D1–D2; being measured by Sweep D |
| Eq 40 contact-mode guards | implemented; enforced at eval, deliberately off during training (D5) |
| sec 3.1 a second object (T-shape) | **not started.** D1 exists so this is possible at all |
| sec 5.2 a flat baseline | **not definable yet** — on a single-edge task the flat arm *is* the skill. Needs the composed task first. A finding, not a delay |
| sec 7 composition (push → recontact) | unblocked, not started |
| sec 9 algorithm independence (PPO) | built and launch-ready; deferred as a replication that gates no decision |
| Stage 2 domain randomization | out of scope by design; friction and mass are fixed |

---

## 4. Open questions, ranked

1. **Build the Stage 1 ladder.** The critical path. Needs no GPU. Everything in
   §3 marked "not yet fitted for contact" is downstream of it.
2. **Build the edge descriptors on orientation, not distance** — Sweep B showed
   distance no longer varies, and a success model needs something that does.
3. **Read Sweeps C and D against their preregistered verdicts.** Written before
   the numbers land, in each launcher's header.
4. **The flat baseline and the composed task are one piece of work.** Fairness
   (sec 5.2) cannot be committed to until the composed board exists.
5. **Second budget rung for Stage 0** — the cheapest remaining new finding,
   costs CPU only, no training.

Full list with next actions: `docs/TODO.md`.
