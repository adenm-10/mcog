#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=sweepC
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-06:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# SWEEP C -- THE INTERFACE DECISION, ON A HARDER BOARD.
# 12 cells = 4 arms x seed{0,1,2} at 2.4M steps, ckpt_freq=600000 (four rungs).
#
# WHAT THIS SWEEP DECIDES, and it is the last thing decided before the ladder:
# the action space, the Gamma encoding, and the observation version that every
# downstream component inherits.
#
# | arm    | action interface | Gamma xi | obs | question                        |
# |--------|------------------|----------|-----|---------------------------------|
# | ctl    | contact_frame    | face     | v2  | does push learn board v2 at all |
# | count  | contact_frame    | COUNT    | v2  | D1: can the face label be retired |
# | raw    | finger_velocity  | face     | v2  | the action-space question       |
# | obs_v1 | contact_frame    | face     | v1  | prices obs v2 against the control |
#
# WHY EVERY ARM RUNS OBS V2, WITH obs_v1 AS THE CONTRAST RATHER THAN THE BASE.
# obs v2 is the version the project intends to keep -- the shared state+xi head
# that composition needs requires it -- so measuring the interface arms at obs v1
# would mean re-running all of them the moment it is adopted. Running them at v2
# and pricing v2 against the control costs the same twelve cells and leaves
# nothing to redo. Sweep B's obsv2 arm CONFOUNDED obs_version with
# normalize_goal_keys and no cell has ever trained obs_version=2 alone.
#
# normalize_goal_keys=false ON EVERY ARM, and that is a finding, not a default.
# VecNormalize keeps a SEPARATE RunningMeanStd per key, but achieved_goal and
# desired_goal are the same physical quantity measured twice, and the whole task
# is making one equal the other. Measured on Sweep B's saved statistics, all
# three seeds: std(achieved)/std(desired) is 1.10-1.14 on the position dims and
# 1.70-1.80 on cos(theta). So a PERFECTLY achieved heading still shows the
# network a residual of up to 1.24 in normalized units, and the residual CHANGES
# SIGN inside the goal window. It also whitens a unit-vector pair, which
# train_contact's own comment forbids for exactly this reason -- the rule was
# written down and then applied to the wrong slice. This is also the first
# account of why obs v2 disagreed between templates: recontact's goal key is
# 2-D fingertip position with NO (cos, sin) pair, so there was nothing to
# corrupt, and obs v2 measured free there.
#
# THE BOARD. slurm/pins_board_v2.sh, sourced not retyped. Full START-pose
# randomization (object_theta_spread_deg=180), goals in the far room
# (portal_goal=false), 3 rooms with doors offset 34cm, require_settled=true,
# cone 75 from a measured p90 of 74.2deg.
#
# TWO REPORTED METRICS, PINNED BEFORE THE SWEEP AND NOT RE-LITIGATED AFTER:
#   (1) mean success on goals >=3cm, POSE reached, under BOTH model and
#       model_best, at each rung. This is the training objective and the number
#       that ranks the arms.
#   (2) on CROSSING episodes only, the fraction that entered the destination
#       region -- "left through the correct portal", which is what a crossing
#       edge inside a route actually has to do. Reported BESIDE (1), never
#       pooled with it: they are different predicates with different floors.
#
# Floors, regenerated 2026-09-08 by tools/make_board_v2_floor.sh BEFORE this
# sweep, split by stratum with tools/split_floor.py:
#
#   arm      digest        pose >=3cm   entered-dst (crossing only)
#   ctl      98e24e99890f     0.000        0.000
#   count    b99150951ac7     0.000        0.000
#   raw      98e24e99890f     0.000        0.000
#   obs_v1   98e24e99890f     0.000        0.162
#
# THE OFFSET DOOR EARNED ITS KEEP HERE. The crossing floor was expected to be
# the problem -- on board v1 a random policy shoved the object through an
# open doorway for 0.271 overall / 0.42 in the crossing bins. Offsetting the
# doors 34cm in y drops it to 0.000-0.162, so the crossing metric is readable
# after all. That was measured, not assumed.
#
# `count` CARRIES ITS OWN DIGEST (b99150951ac7) because guard_contact_count is a
# TASK key -- enforcing a count is a different task from being told one. So it
# gets the two-way treatment: scored on its own protocol AND zero-shot on the
# common one, because the DELTA against ctl is the quantity of interest. The
# other three arms differ only in INTERFACE keys and share 98e24e99890f.
#
# PREREGISTERED VERDICTS, written before the numbers land:
#   ctl clears Bar 1's SHAPE on board v2 (pose >=3cm >= 0.40, on >=2 of 3 seeds,
#     under BOTH checkpoints, no distance bin at 0.00) or THE BOARD IS WRONG and
#     no arm is readable. Stop and fix the board rather than reading an arm.
#   count is ADOPTED iff the paired per-episode bootstrap CI on (count - ctl) has
#     an upper bound >= -0.15. Deliberately wider than raw's: it buys an
#     abstraction that survives a T-shape, a round object and 3D, and needs only
#     tactile sensing rather than face localization, so it is worth real
#     performance.
#   raw is ADOPTED iff (raw - ctl) upper bound >= -0.05. It removes an
#     assumption, so it wins ties.
#   obs v2 is ADOPTED iff (ctl - obs_v1) lower bound >= -0.05, i.e. v2 is not
#     materially worse than v1 at the control interface. If v2 LOSES, say so and
#     price it; do not quietly keep both.
#
# HOW TO READ IT, and Sweep B is why these are stated rather than assumed:
#   - FOUR RUNGS, and the deliverable is the CURVE. Sweep B's arm ordering
#     flipped between 1.8M and 2.4M (widecone went 2nd -> 3rd -> tied -> 1st), so
#     an adoption decision must hold at 1.8M AND 2.4M or it is not a decision.
#   - BOTH CHECKPOINTS, and disagreement is itself a verdict. Sweep B's widecone
#     was +0.118 on model and exactly +0.000 on model_best; an arm that splits
#     like that has not converged and defers to the curve.
#   - THE BENCHMARK'S RESOLUTION IS ONE EPISODE IN 72. Two Sweep A/B checkpoints
#     that were bit-identical in weights to 9e-4 differed on 3-7 of 48 episodes.
#     Report the point estimate beside the CI bound.
#   - EXPECT BOARD V2 TO STILL BE CLIMBING AT 2.4M. Every harder arm in Sweep B
#     was, and this board is harder than all of them.
#   - CALIBRATION READINESS, free, from the evals already produced: report the
#     winner's distance-bin spread and its orientation split (inside tol vs
#     must-rotate). If BOTH are under 0.10, push has gone flat again and the
#     Stage 1 ladder calibrates on an earlier rung or a harder protocol rather
#     than on the converged winner. Sweep B's ctl went 0.278 -> 0.194 -> 0.083
#     across its rungs, which is why this is a preregistered read-out.
#
# NOT IN THIS SWEEP, each omission a measurement rather than a budget cut:
#   raw_count   DEFERRED to a conditional 3 cells, run only if BOTH count and
#               raw are adopted -- the only branch where the interaction matters.
#   guard_face  D5: OFF here and re-scored zero-shot on all 12 cells. `adjacent`
#               costs 0.042-0.083 and IS enforceable, but an interface
#               comparison between two floored arms answers nothing.
#   PPO         answers memo Table 4's algorithm-independence REPLICATION and
#               gates no design decision. The floor branch is done and it is
#               launch-ready whenever the task is frozen.
#   the flat baseline  still not definable: on a SINGLE-EDGE task memo sec 5.2's
#               "identical reset distribution and action space, no temporal
#               hierarchy" IS the push option. Empty until a composed task exists.
#
# BUDGET AND WALL. 2.4M at Sweep B's MEASURED 18.3h/cell median (range
# 15.1-20.3) against a 30h wall. ckpt_freq=600000 puts snapshots at 600k/1.2M/
# 1.8M on disk, so a cell killed at the wall still yields three rungs.
# Eq 35 accounting, MEASURED rather than quoted as the training budget alone:
# the diagnostic eval (32 full-task + 32 local episodes every 5000 steps) is not
# free. On smoke 45449813 it consumed 2.09x the TRAINING steps, so a 2.4M cell
# costs ~7.4M interactions, not 2.4M. Sweep C total ~89M environment
# interactions (12 x 2.4M train + ~12 x 5.0M eval), plus 4 untrained floors at
# zero gradient steps. Report the 89M, not the 28.8M (CLAUDE.md, Reporting).
# ===========================================================================

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/pins_board_v2.sh

ARMS=(ctl count raw obs_v1)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# Held FIXED across all four arms, and stated rather than defaulted: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
OBS="obs_version=2 rich_obs=true normalize_goal_keys=false"
ACT="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 \
gap_assist=false mask_inactive_finger=true"
XI="xi_gamma_mode=face"
GUARD=""

case "${ARM}" in
  ctl)    ;;
  count)  XI="xi_gamma_mode=count"
          # The WHOLE of D1, not just the label: count xi, the count guard, and
          # the second finger free so "how many contacts" is a real choice.
          GUARD="guard_contact_count=1"
          ACT="action_interface=contact_frame slip_model=speed_fraction \
slip_limit=1.0 gap_assist=false mask_inactive_finger=false" ;;
  raw)    ACT="action_interface=finger_velocity slip_model=speed_fraction \
slip_limit=1.0 gap_assist=false mask_inactive_finger=true" ;;
  obs_v1) OBS="obs_version=1 rich_obs=true normalize_goal_keys=false" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=2400000
CKPT_FREQ=600000
EVAL_EPS=32   # not 16: the curriculum advance gate compares against 0.4, and
              # 16 episodes make that a 6-of-16 coin flip.

EXTRA_OVERRIDE="${BOARD_V2_TRAIN_PINS} ${OBS} ${ACT} ${XI} ${GUARD} \
ckpt_freq=${CKPT_FREQ} diag_eval_episodes=${EVAL_EPS}"
RUN_TAG="push_${ARM}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"
# THE BENCHMARK PROTOCOL, written beside the runs rather than retyped in the
# scorer -- a protocol that lives only in a launcher comment is what made every
# v25 cross-version comparison wrong. This is the BENCHMARK form
# (curriculum_levels=null, the reverse sampler at full range, which is what
# every arm's last curriculum level trains on), not the training form.
# Written via mv, which is atomic: 12 tasks race here with identical content.
printf '%s\n' "${BOARD_V2_PINS}" > "${SWEEP_DIR}/.PINS.$$"
mv -f "${SWEEP_DIR}/.PINS.$$" "${SWEEP_DIR}/PINS.txt"

GROUP=sweepC
FINALIZE_TAG=sweepC
RUNG_SCORER=tools/score_rungs.sh
source slurm/_run_cell.sh
