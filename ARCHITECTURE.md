# MCOG — architecture

**Who this is for: a human.** It explains the shape of the repo, how information
flows through it, and which four pieces everything else rests on. It carries no
measured numbers and no protocol strings — those live in `docs/PROGRESS.md` and
`slurm/pins_*.sh`, and copying them here is how this project's worst bugs start.

Dense references, for a Claude session or a deep dive:
`status.md` (current state + gotchas) · `docs/PROGRESS.md` (dated log) ·
`docs/TODO.md` (open work) · `docs/STRUCTURE.md` (file-by-file map) ·
`RESEARCH_LOG.md` (what each experiment answered).

---

## 1. What this repo does

A robot has one simple skill: push a block across a table with a fingertip.
That skill works from some starting positions and fails from others.

This repo does four things:

1. **Trains** the skill (`train_contact.py`).
2. **Measures** how reliably it works from many different starting states
   (`eval_contact.py`, `option_graph/calibrate.py`).
3. **Fits a model** of "given where I'm starting, will this skill succeed?"
   (`option_graph/edge_model.py` — the memo calls this `p_hat`).
4. **Plans a multi-step route** using that model, runs it, and checks whether
   the prediction held (`option_graph/planner.py` -> `option_graph/executor.py` -> `option_graph/metrics.py`).

Step 4 is the research claim: that knowing *where a skill fails* lets you plan
better than assuming it always works.

---

## 2. The map

```
  ┌────────────────────────────────────────────────────────────────────┐
  │  WHAT YOU RUN                                                      │
  │    train_contact.py  ──▶  a checkpoint  ──▶  eval_contact.py       │
  │    (learn one skill)      (.zip)             (the only number that │
  │                                               compares across runs)│
  │    configured by Hydra:  config/                                   │
  └────────────────────────────────────────────────────────────────────┘
                                  │
  ══════════════════ THE CONTRACT — five functions ═══════════════════════
        region_of        "which room am I in?"
        resolve_target   "where am I trying to get to?"
        guard_cells      "where am I allowed to be?"
        guard            "did I just break a rule?"
        score_arrival    "did I get there?"
     (option_graph/executor.py :: DomainHooks)
  ════════════════════════════════════════════════════════════════════════
                     ╱                              ╲
  ┌──────────────────────────┐        ┌──────────────────────────────────┐
  │ domains/nav/             │        │ domains/contact/                 │
  │ STAGE 0 — finished       │        │ STAGE 1 — active                 │
  │ a car driving a maze     │        │ two fingers pushing a block      │
  │                          │        │   world.py  ← the only file that │
  │ the worked example of    │        │              imports pymunk      │
  │ the contract; keep it    │        │   gym_env · reward · her_buffer  │
  └──────────────────────────┘        └──────────────────────────────────┘
                     ╲                              ╱
                      domains/geometry.py
                      domains/contact_templates.py
                      (what both domains genuinely share)

  ┌────────────────────────────────────────────────────────────────────┐
  │  option_graph/ — THE CORE. Imports no environment, ever.           │
  │                                                                    │
  │    executor.py  ──writes──▶  records.py  ──read by──▶ calibrate    │
  │    (the only thing           (the only shared         edge_model   │
  │     that makes records)       vocabulary)             metrics      │
  │                                                       planner      │
  └────────────────────────────────────────────────────────────────────┘
```

**The one idea to hold onto: the core cannot see the physics.** `option_graph/`
never imports an environment. It reads only *records* — "I tried edge X from
state Y and it worked / it failed." That is what lets a maze car and a pushing
finger share one planner, and it is why their results are comparable at all.

A test enforces this. `option_graph` modules are checked to import no
`gymnasium`, no `stable_baselines3`, no `pymunk`, and no domain code.

---

## 3. How information flows

```
   train ──▶ checkpoint ──▶ calibrate ──▶ records ──▶ edge_model ──▶ planner
                            (roll out      (.jsonl)   (fit p_hat)    (pick a
                             a lot, no                                route)
                             learning)                                  │
                                                                        ▼
        metrics  ◀──  records  ◀──  executor  ◀────────────────────────┘
      (was the model    (.jsonl)    (actually run
       right?)                       the route)

   STAGE 0 (maze):    ████████████████████████████████████████  complete
   STAGE 1 (contact): ██████████                                stops here
                                └── contact/hooks.py implements the contract,
                                    and NOTHING CALLS IT. See §5.
```

### One episode, end to end

1. `ContactEnv.reset()` samples a start: where the block is, where the fingers
   are, where the goal is. Which sampler runs depends on `curriculum_mode`.
2. `domains/contact/world.py` steps pymunk 20 times per policy tick (500 Hz physics, 25 Hz
   policy) and returns the raw state.
3. `obs()` turns that state into what the policy sees. It is *object-centric*:
   shift the whole scene 30 cm and the observation is unchanged. That is what
   lets one policy work anywhere on the board.
4. The policy acts; `guard` checks no rule broke; `score_arrival` checks
   whether the goal was reached.
5. At episode end the outcome becomes an `EpisodeRecord` — the one wire format
   every downstream analysis reads.
6. `edge_model` fits `p_hat` over many such records; `metrics` scores whether
   `p_hat` predicted composition better than the naive alternatives.

### Two words that unlock most of the docs

**env digest** — a hash of the task definition (`98e24e99890f`, and so on).
Two numbers are comparable **only** if their digests match. When you see a hash
next to a result, it is a receipt saying "measured on the same task."

**INTERFACE vs TASK keys** — the most important distinction in the project.

| | changes | in the digest? | so… |
|---|---|---|---|
| **INTERFACE** | what the robot's controls *mean* | **no** | two control schemes compare on ONE benchmark |
| **TASK** | what success *is*, or which situations occur | **yes** | changing one creates a NEW experiment |

`domains/contact/keys.py` holds the one list. Getting a key on the wrong side
silently makes two incomparable numbers look comparable.

---

## 4. The four backbones

### 1. `DomainHooks` — the abstraction boundary
Five callables (§2) are the *entire* surface a domain must implement. A new
object, skill, or simulator is a new file under `domains/` implementing them;
`option_graph/` does not change. Its own rule: **widen `resolve_target` before
adding a sixth hook.**
*Without it:* every new domain forks the planner, and no two results compare.

### 2. `option_graph/records.py` — one vocabulary, one serializer
The only shared data format, and the only JSON writer in the repo.
*Without it:* five near-copies of the serializer existed once and had already
diverged on how they handled numpy arrays. A test now fails if a second
definition appears anywhere.

### 3. `domains/contact/keys.py` — the INTERFACE / TASK split
The list that decides which numbers may be compared. It is deliberately
**dependency-free**, so even a stdlib-only script can import it instead of
copying it.
*Without it:* five copy-pasted copies, kept in sync by a test that could only
fire *after* one drifted.

### 4. `pins_*.sh` — one definition of a task protocol
`slurm/pins_board_v2.sh` is sourced by the launcher, the floor builder and the
scorer. Never retyped.
*Without it:* a protocol was retyped with one key missing, and the "untrained
baseline" was measured on a different task than the policies it bounded.

**The pattern behind all four:** the same quantity computed in two places, with
nothing forcing them to agree. That is the source of roughly twenty-two defects
in this project. Every fix that stuck has the same shape — **collapse the
copies, then add a test against regrowth.**

---

## 5. Extending it

**Adding a new object, skill, or simulator** — implement the five hooks in a new
file under `domains/`. Checklist:

| you want to | you touch |
|---|---|
| a new physics substrate (e.g. a MuJoCo arm) | the top half of `domains/contact/world.py`'s pattern — a new sibling file. The bottom half (`obs`/`step`) is reused |
| a new skill (pivot, pinch) | `domains/contact_templates.py` + a `gym_env` mode |
| a new board | `slurm/pins_*.sh` — a new pins file, never an edit to an old one |
| a new sweep | a launcher + `tools/score.sh` variant |
| how a number is scored | `eval_contact.py` — and expect every stored score to need re-deriving |

**Rules that are not negotiable**, each paid for:
- Widen an existing config key before adding a new one. A new env kwarg
  re-hashes every digest and orphans every stored score.
- Deprecate, never delete, an interface that archived checkpoints trained on.
- Anything sampled at `reset` is re-read every episode. A cached per-episode
  value is silent corruption, not a crash.

### The one thing blocking scale

`domains/contact/hooks.py` implements the contract for the contact domain, and
**nothing calls it.** Not because it is wrong — because the machinery below the
contract is still maze-shaped: it expects grid cells and counts hops. Until
there is a contact version of *"which states can this skill start from"* and
*"how do I describe an edge numerically"*, the right half of the loop in §3
cannot run on contact.

That is the Stage 1 ladder. It needs no GPU. It is the difference between a
two-domain system and a one-domain system with a second training pipeline
bolted to its left half.
