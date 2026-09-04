# MCOG

Hierarchical RL + graph planning for contact-rich manipulation: a graph of state-space
regions, set-to-set options with contact templates, learned edge-success models, risk-aware
planning. Python 3.11, conda env `tsmc`, run from repo root.

**Read before acting:** `status.md` (current state, gotchas, reference numbers) ->
`docs/TODO.md` (open work, ORDER OF WORK first) -> `docs/STRUCTURE.md` (repo map, layering,
gates). Those files hold what is true *today*; this one holds what is true every time.
A handoff doc's "confirmed on disk" is a claim, not a fact — verify by reading the file.

## The memo is the reference

`docs/hierarchical_contact_rich_manipulation_research_plan.pdf` is the spec. Cite its
section/equation numbers for anything that traces to it (Eq 7 `xi_e`, Eq 10 `A(v,e)`,
Eq 22 `p_hat`, Eq 29 `H`, Eq 31/32 cost, Eq 35 budget, sec 5.2 baselines, sec 5.4 splits).

**It is authoritative until one of our own measurements overrides it — and then the
deviation is recorded with its measured price.** Never deviate to make a number look better,
and never deviate without the measurement that justifies it.

## Scientific integrity — the top priority

- **Verify the env digest before quoting any number.** Two success rates measured on
  different reset distributions are not comparable, however carefully each was computed.
  A cell's own training-time eval is not a cross-cell number; use the shared benchmark.
- **When a digest moves, find out why before scoring.** Adding a key to the stamp changes
  the hash while the task stays identical; changing a task key's value changes the states.
  Separate them by diffing the per-episode initial states, not by reading the hash.
- **INTERFACE vs TASK keys.** INTERFACE changes what the policy's outputs mean: read it per
  cell from the run's own metadata and keep it out of the digest. TASK changes what success
  is, or which states are visited: pin it at one value for every arm, inside the digest.
  **Two arms differing in a task key are two experiments** and need a transfer eval before
  they can be compared.
- **An eval protocol's task keys live beside the numbers**, not in a launcher comment.
- **Regenerate the untrained floor for every protocol change, before the sweep.** A floor is
  specific to (interface, goal space, protocol) and never transfers.
- **Before crediting a change, check the launched run's recorded config for the key**, not
  the code for the fix. A flag no launcher sets is not a feature.
- **Count factors against arms before launching.** Three arms cannot bisect eight changes,
  however well justified each one was on its own.
- **Score every checkpoint a claim rests on** (final and best-so-far); if they disagree, the
  runs have not converged.
- **Commit before submitting a sweep**, or provenance is a guess.
- **Every number carries its source** — digest, job id, artifact path. Report negative and
  null results as prominently as positive ones.

**Spend the cheap check before the GPU.**

- Print the distribution a task-design change produces against the distribution with it off.
- Ask whether a frozen checkpoint already answers the question. Valid for changes the
  policy's outputs don't depend on; not for changes to the action space or the observation.
- For any goal or arrival change, construct the state that perfectly satisfies the goal and
  assert the env calls it a success.
- Any claim of the form "the physics prevents X" needs the measurement, not the derivation.
- Ask what a statistic *could* have read before quoting it as evidence — a threshold that
  ends an episode cannot also measure failures against itself.

**Reporting.**

- **Report total environment interactions across every component (Eq 35)**, not the largest
  one. Calibration and edge-model data count.
- **Fairness (sec 5.2/7): a baseline inherits the identical reset distribution and action
  space**, or a hierarchy win is a curriculum artifact. Any reset-distribution change forces
  a re-baseline.
- **A benchmark's primary metric is pinned in writing before the sweep and reported
  unchanged afterwards**, alongside any restriction it applies.
- **Success bars are written down before they are needed, and not re-litigated afterwards.**
  The current ones are in `docs/TODO.md`.

## Testing

Gates run before and after every change; the authoritative list and counts are in
`docs/STRUCTURE.md`. Today:

```bash
python test_code.py static
python test_code.py geometry
python test_code.py contact
python -m tests.test_option_graph all
python -m tests.fixture_eval fixtures tests/fixtures_smoke
```

- Hand-rolled harness (`check`/`section`/`report`). **No pytest in this repo.**
- **Never report a gate as passing that you did not run**, including one whose tool isn't
  installed in the env.
- **Assert the branch actually fired**, not just that the output matched.
- **Assert on content, never on an artifact existing** — for plots, on the artists
  (`ax.lines`, `ax.collections`, the title). A valid empty PNG appears every run.
- **A new env mode or goal space needs an end-to-end check**, not only unit coverage.
- **Deleting one side of an equivalence test kills the test.** Replace it with an
  independent oracle rather than a diff of a function against itself.
- Smoke-test new diagnostics and launchers on real (even tiny) data, driving the real file:
  a retyped copy of a launcher is a different experiment.

## Consistency

- **One definition per concept, repo-wide.** Shared vocabulary and serialization have a
  single home; a second copy diverges silently and `static` fails on it.
- **Wire formats are frozen unless a change regenerates every record.** Record loading
  drops unknown keys silently, so a renamed field fails with no error.
- **Names follow the memo** (`p_hat`, `p_bar`, `H`, `A`, `beta`, `xi_e`, `predictor`,
  `EntryCondition`, `Route`). If one term means two things, rename per-file, not by sed.
- **Hold definitions, units, seeds and metrics fixed within a session**; announce changes.
  Record both clocks: the derived one and the experimental one layered on top of it.
- Write in the idiom of the surrounding file — its naming, comment density, and structure.

## Scalability

The memo's later stages add templates, substrates, baselines, and randomization axes. Write
for that without pre-building it.

- **The core stays domain-agnostic and env-free.** A new domain or template is added behind
  the same boundary the current ones use — by implementing the existing hooks, not by
  special-casing the core or an older domain — so their numbers stay comparable.
- **Widen an existing config key before adding a new one.** A new env kwarg rehashes every
  config and orphans every stored score. New options default to bit-identical old behaviour
  (same RNG draw count and order) and ship with a gate check.
- **Deprecate, don't delete, an interface that archived checkpoints were trained on.**
- **Keep physics and geometry as data, not constants** — held-out geometry and dynamics
  randomization both read them per episode.
- **Anything sampled at `reset` is re-read every episode**; a cached per-episode value is
  silent corruption, not a crash.
- **A constant with a closed form is derived, not swept** — and never model the same physics
  twice; check whether the solver already does it.

## Environment

```bash
module load python && mamba activate tsmc     # own unpiped line, or python resolves to base
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"   # else numpy GLIBCXX crash
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
```

The env is `tsmc`, not `to-smc` as `environment.yml` says. Every entry point uses real
Hydra: `key=value`, dash -> underscore, no `--flags`.

- **`grep -r` here honours `.gitignore`, which excludes `tests/*`** — use `command grep`
  before calling anything unreferenced.
- **The login node has one core.** Anything parallel goes to a compute node via `sbatch`.
- **Progress is the training CSV, never a log file's size** (stdout is block-buffered).
- **Hydra list-of-dict overrides must reach the CLI through a shell variable** (bash
  brace-expands `[{a,b}]`).
- **wandb: no media, ever** — plots and video to local disk. `WANDB_API_KEY` is in
  `~/.bashrc`, so a non-login shell needs `source ~/.bashrc` first. Project `mcog`.

## Docs — update in the same change that invalidates them

`status.md` (state + gotchas, the entry point) · `docs/TODO.md` (next action each) ·
`docs/PROGRESS.md` (dated: question -> what ran -> result -> next) · `docs/STRUCTURE.md`
(tree, layering, gates). A gotcha earns its place by having cost something: write the rule,
not the story.

## Code quality — the standard is a staff ML/robotics engineer

The minimum clear code that solves the problem. A staff reviewer should find nothing to cut,
nothing to rename, and no comment that a better name would have replaced.

- **Take the cleanup when you see it**, in the same change: delete dead code, halve a
  function that can be halved, replace a *what* comment with a clearer name, collapse an
  abstraction with one caller. Leaving it is a choice too. This overrides the general
  "surgical changes, don't touch adjacent code" rule — here, drive-by cleanup is wanted.
- **Prefer deleting to adding.** No speculative abstraction, configurability, or error
  handling nobody asked for. If 200 lines could be 50, write the 50.
- Docstrings <=3 lines/function, <=7/module, and only where the signature isn't
  self-explanatory. Inline comments sparse and why-only.
- **Simplify from understanding, not from pattern-matching.** If you can't say what the code
  you are cutting was for, you are not ready to cut it.
- Keep the cleanup legible as a separate concern from the behaviour change, so a reviewer can
  tell which lines could have changed a number.

Three things this project has actually paid for, so they bound the mandate:

- **Prove dead code is dead first.** Recursive grep here honours `.gitignore` (which excludes
  `tests/*`), relative imports defeat naive import scans, and `# noqa: F401` re-export
  façades look unused on purpose. Two wrong deletion calls came from exactly this.
- **A behavioural no-op still moves the diff hash.** Land deletions between sweeps, never
  during one, and never remove an interface that archived checkpoints were trained on.
- **A cleanup must leave the gates green and the numbers bit-identical.** If the code you
  touched writes an artifact, regenerate it and diff.
