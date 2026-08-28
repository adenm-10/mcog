#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=push_lean
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v30 FAMILY A -- the config gate. 6 cells = {lean, lean_raw} x seed {0,1,2},
# 1.2M steps each (~8h/cell measured; 400k took 2.4-3.0h in job 42300917).
#
# THE QUESTION. v29 (job 42300917) costed five scaffolds single-factor against
# v28's `full` on one 60-episode benchmark, and four came back free or POSITIVE:
#
#   nogapassist  0.822  +0.083 paired, 3/3 seeds, McNemar p=0.017  <- BETTER
#   physdamp     0.789  +0.050, holds under BOTH checkpoints
#   unmask       0.756  +0.017, p=0.775 (a wash on success)
#   randtheta    0.694  -0.044, p=0.322 -- but 0.717 on its OWN task against
#                       `full` transferred in at 0.578, so training recovers it
#   rawact       0.217  -0.522   <- the ONLY load-bearing scaffold
#
# Each of those four was measured ALONE. `lean` runs all four together for the
# first time. Four individually-free changes are not automatically jointly
# free, and two of them (`unmask`, `randtheta`) had the STEEPEST slopes in the
# whole sweep (+0.148 and +0.146 per 100k over the last 150k), so their "wash"
# verdicts were budget-limited rather than converged. 30 of 42 v29 cells were
# still rising at their cap (pooled sign test p=0.008).
#
# `lean_raw` is the same config with the raw finger-velocity action space, i.e.
# the one scaffold v29 proved load-bearing, removed. It is the control that
# says how much of `lean` is the contact frame.
#
# WHY THIS GATES EVERYTHING ELSE. If `lean` lands well below `physdamp`'s 0.789
# then the four changes interact, and any downstream geometry or goal-diversity
# number measured on top of `lean` is confounded. Do not launch the diversity
# sweep until this is scored.
#
# ---------------------------------------------------------------------------
# ckpt_freq=400000 IS LOAD-BEARING HERE, NOT A CONVENIENCE.
#
# The comparison that answers the question is `lean` against v29's
# `nogapassist`/`physdamp`, and those ran at 400k. Scoring lean's 1.2M endpoint
# against a 400k baseline would confound the config change with 3x the budget.
# The 400k snapshot makes it budget-matched on the same 60-episode benchmark.
# It also gives 3 points per run toward memo Eq 37/38 (N_rho and AULC, both
# currently nan and never used).
#
# Cost: 3 extra snapshots x 3.26 MB x 6 cells = ~59 MB. PRUNE AFTER SCORING --
# periodic checkpointing is exactly what tools/prune_runs.py exists to contain,
# which is why config/train_contact.yaml defaults ckpt_freq to null.
# ---------------------------------------------------------------------------
#
# WHAT `lean` KEEPS, and why none of these is a "trick":
#   action_interface=contact_frame  the one thing v29 proved load-bearing. The
#                                   policy's two live outputs are read as (push,
#                                   slide) in the contacted face's frame,
#                                   re-derived every physics substep, instead of
#                                   raw (vx, vy). Under the raw space, steering
#                                   requires tangential motion, which slides the
#                                   finger off the face -- aiming and holding
#                                   contact are in conflict. In the contact
#                                   frame "push" cannot break contact by
#                                   construction, so they come apart.
#   disengaged_away_deg=60          with mask_inactive_finger=false this is the
#                                   only thing preventing own-finger collisions
#                                   (forbidden_contact 2% -> 8-11% at `unmask`,
#                                   against v27's 26.7-31.7% WITHOUT it). Costs
#                                   nothing and removes a failure mode that is
#                                   not the research question.
#   push_range_min_cm=3.0           KEPT, reversing an earlier call of mine. v28
#                                   measured the floor's effect on benchmark
#                                   success at exactly 0.00, so it first looked
#                                   like "one less trick". Measured consequence
#                                   with it OFF: the same-room training goal
#                                   median is 2.3cm against a benchmark median
#                                   of 7.2cm, i.e. most training episodes need
#                                   under 1cm of object motion and are
#                                   near-trivial (success under 1cm is 0.856).
#                                   With it on the median is 4.5cm (measured,
#                                   tools/probe_scaling_geometry.py). Neutral on
#                                   score, saner distribution, so it stays.
#                                   Matching the benchmark's 7.2cm would want a
#                                   floor near 5-6cm, which is UNTESTED.
#   require_settled=false           UNCHANGED from v28/v29 so this sweep stays
#                                   comparable. Turning it on is the separate
#                                   change that makes push composable
#                                   (docs/TODO.md Immediate 7) and it moves the
#                                   env digest, so it must not ride along here.
#
# WHAT `lean` DROPS relative to v28's `full`:
#   gap_assist        -> false   v29: removing it HELPED (+0.083, p=0.017)
#   mask_inactive     -> false   v29: wash on success, worse on collisions
#   theta_spread      -> 90      v29: free. Object heading was 0 in 300/300
#                                resets before; now median |heading| 44.7deg
#                                (measured over 400 resets).
#   angular_drag_arm  -> 3.12    the DERIVED value. tau = mu*m*g*L and L is the
#                                pressure-weighted mean radius of the contact
#                                patch: 3.12cm for a uniform 10x6, with 5.83cm
#                                (the half-diagonal) the hard ceiling for ANY
#                                pressure distribution. The shipped 6.00 was
#                                ABOVE that ceiling. NOT 3.14 -- that is pi and
#                                the resemblance is a coincidence.
#   push_cone_deg     -> 90      Phase 0 costed 30->90 at ~0.02, and a v30
#                                replay confirms it: nogapassist 0.822 -> 0.767,
#                                physdamp 0.733 -> 0.733 (logs/eval/v30_conesweep).
#                                NOT null: push_cone_deg=null does NOT widen the
#                                goal distribution, it drops into the historical
#                                sampler whose face choice comes from room-centre
#                                heuristics (finger lands on the goal side in
#                                40-57% of resets, v20), AND it raises
#                                ValueError with object_theta_spread_deg
#                                (gym_env.py:167). Widening means 180, not null.
#
# NOT IN THIS SWEEP, deliberately:
#   Portal and room size. A v30 probe killed that axis: the cross-room sampler
#   REQUIRES the goal's ray to pass through the portal, so a straight path
#   exists in 93-100% of cross-room episodes even at a 6.5cm gap (a gap barely
#   wider than the 6cm object), and goal misalignment stays at ~16deg for EVERY
#   cone width cross-room. Narrowing the portal tightens aim; it never forces
#   the object around an obstacle. 18 cells / 144 GPU-hours cut before running.
#   Board size additionally cannot share a benchmark at all: SB3's
#   check_for_correct_spaces compares the saved goal Box bounds
#   [board_w, board_h], so a 90x54 checkpoint raises ValueError against 50x30
#   (all 12 of v29 `bigroom`'s cross-board evals failed exactly this way).
#
#   Goal diversity. This IS the live axis -- 30deg -> 180deg costs 0.26
#   (nogapassist 0.822 -> 0.561) and at 180deg 51.7% of same-room goals sit
#   BEHIND the contacted face. But it is a separate sweep and it is gated on
#   this one, because it has to be measured on top of a config we trust.
#
# SCORING. One digest group, so one command:
#   sbatch slurm/score_sweep.sh logs/sweep_<JOBID> logs/eval/v30_lean_sameroom
#   which pins tools/score_sweep.py's TASK_PINS: the SAME 60-episode benchmark
#   (digest daee708c3fa6) that v27/v28/v29 were scored on, so these numbers drop
#   straight into that table. INTERFACE keys come from each cell's meta.txt, so
#   lean_raw is read as (vx, vy) and lean as (push, slide) -- getting that
#   backwards inverted a whole result in v25.
#   Then the budget-matched row against v29, from the 400k snapshot:
#     python tools/score_sweep.py logs/sweep_<JOBID> \
#       --out-dir logs/eval/v30_lean_400k --ckpt model_400000_steps.zip
#   SB3's CheckpointCallback names them model_<step>_steps.zip -- verified by
#   smoke test, NOT model_400000.zip.
#   Floor for that benchmark already exists (logs/eval/v29_floor/): untrained
#   contact_frame 0.150 all / 0.000 on goals >=3cm, untrained finger_velocity
#   0.067 / 0.021. REPORT >=3cm AS THE PRIMARY METRIC -- the 5-bin mean has a
#   0.150 floor coming entirely from the 0-3cm bin.
# ===========================================================================

set -e

ARMS=(lean lean_raw)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# portals=[{...}] must reach Hydra through a shell variable: bash brace-expands
# [{a,b,c}] into three words, in heredocs and sbatch scripts alike.
PORT="portals=[{x:25.0,y_lo:5.0,y_hi:25.0}]"

# Stated explicitly in every arm, never left to a default: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 mask_inactive_finger=false gap_assist=false"
TASK="push_cone_deg=90 require_settled=false disengaged_away_deg=60 push_range_min_cm=3.0 object_theta_spread_deg=90 angular_drag_arm_cm=3.12 board_w_cm=50.0 board_h_cm=30.0"
SRG="same_room_goal_prob=1.0"

case "${ARM}" in
  lean)     ;;                                   # base, unmodified
  # gap_assist and slip_* are contact_frame-only keys, so the raw arm drops
  # them rather than setting them false: finger_velocity has never had the gap
  # assist, which is why `false` is the midpoint of full -> raw and not raw.
  lean_raw) IFACE="action_interface=finger_velocity mask_inactive_finger=false" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=1200000
CKPT_FREQ=400000
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true ${PORT} min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 target_clip=10 ckpt_freq=${CKPT_FREQ} ${SRG} ${TASK} ${IFACE}"
RUN_TAG="push_${ARM}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "TEMPLATE=${TEMPLATE} TOTAL_STEPS=${TOTAL_STEPS} EXTRA_OVERRIDE=${EXTRA_OVERRIDE}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)";
  # GIT_COMMIT alone could not tell job 40944664 (pre-fix) from 40957220
  # (post-fix): both recorded a7c153a because the fix was uncommitted. Record
  # whether the tree was dirty and a hash of the diff, so provenance is
  # checkable rather than a guess (docs/PROGRESS.md, v20).
  echo "GIT_DIRTY=$(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)";
  echo "GIT_DIFF_SHA=$(git diff HEAD 2>/dev/null | sha256sum | cut -c1-16)"; } > "${RUN_DIR}/meta.txt"
git diff HEAD > "${RUN_DIR}/uncommitted.diff" 2>/dev/null
cp "$0" "${RUN_DIR}/submit_script.sh"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_contact.py \
  contact=${TEMPLATE} total_steps=${TOTAL_STEPS} seed=${SEED} \
  ${EXTRA_OVERRIDE} \
  out_dir="${RUN_DIR}" \
  wandb=true wandb_run_name="${RUN_TAG}_${SLURM_ARRAY_JOB_ID}" \
  wandb_group="lean_${SLURM_ARRAY_JOB_ID}"
rc=$?

still=$(squeue -h -j "${SLURM_ARRAY_JOB_ID}" -t PENDING,RUNNING -o "%A_%a" \
        | grep -v "^${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}$" | wc -l)
if [ "${still}" -eq 0 ]; then
  mkdir -p "${SWEEP_DIR}/slurm_logs"
  find logs/slurm_staging -maxdepth 1 -type f \
       \( -name "${SLURM_ARRAY_JOB_ID}_*.out" -o -name "${SLURM_ARRAY_JOB_ID}_*.err" \) \
       -exec mv -t "${SWEEP_DIR}/slurm_logs/" {} + 2>/dev/null
fi
exit $rc
