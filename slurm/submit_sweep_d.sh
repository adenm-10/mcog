#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=sweepD
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-18:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# SWEEP D -- THE GAMMA LADDER. Is the contact interface acquirable at all?
# 6 cells = 2 arms x seed{0,1,2} at 1M steps, ckpt_freq=250000 (four rungs).
#
# THE QUESTION. Recontact's job is to move the fingers between canonical contact
# interfaces while the object stays put: the memo's node is v = (C_i, Theta_j,
# Gamma_l) and recontact changes Gamma while holding C and Theta fixed. Eq 13's
# positional form returned 0.000 on 12 of 12 cells. Phase 0 then established
# that this was NOT bad geometry -- fingers placed exactly on target score
# arrived 80/80 for all three classes with the object moving 0.000cm -- and that
# the still-guard was the blocker: `object_disturbed` fired on 237 of 240
# scripted episodes at a median tick 8 of 200, and because the flag is STICKY
# and gates _her_arrived it had switched HER OFF for ~96% of every episode.
#
# So this sweep tests ONE hypothesis: with Gamma read as a CONTACT COUNT (D1)
# and the invariant read as a LATCHED DISPLACEMENT bound rather than an
# instantaneous velocity test (D2), is the interface acquirable?
#
# | arm        | goal      | start          | commanded count | floor |
# |------------|-----------|----------------|-----------------|-------|
# | g_one      | push      | free           | 0 -> 1          | 0.000 |
# | g_two_disp | pivot,pinch | free, push   | 0 or 1 -> 2     | 0.000 |
#
# THE LADDER IS DIRECTIONAL, AND THE FLOORS ARE WHY. Two earlier framings were
# discarded on measurement, both before any GPU:
#   init drawn from all four classes  -> floors 0.583 / 0.396, because roughly
#     half of episodes SPAWN already holding the commanded number of contacts.
#   init drawn from every class whose count differs -> g_one floor still 0.542,
#     because two thirds of those start with TWO contacts and a random policy
#     reaches "one" by simply DRIFTING OFF one of them.
# MEASURED FINDING, reported rather than hidden: RELEASING a contact is not a
# skill (untrained floor 0.542). ACQUIRING one is. So both rungs are
# acquisitions, and the release direction is reported as the 0.542 column it is.
#
# AND A REAL LIMITATION OF THE COUNT ABSTRACTION, also reported: pinch <-> pivot
# is a genuine interface transition (two contacts, different geometry) that
# contact count cannot express, so it reads as a no-op. Count buys transfer to a
# T-shape, a round object and 3D; it pays for it by collapsing same-count
# regrasps. Those need the positional Gamma, which is a separate arm and is
# currently 0.000.
#
# THE FOUR SETTINGS ARE ONE HYPOTHESIS, not four knobs. The probe says they are
# complementary rather than alternatives:
#   guard_object_still=displacement, eps=2cm, LATCHED RUNNING MAX. At a 2cm/s
#     action scale the peak disturbance is a median 0.62-0.76cm (p90 1.03-1.19),
#     so eps=2cm leaves 1.000 of ticks eligible; eps=1cm leaves 0.961-0.979 and
#     is marginal for no benefit. The latch matters: instantaneous net
#     displacement would let a policy shove the object and push it back.
#   v_max_cm_s=2. At 20cm/s even eps=3cm leaves only 0.33-0.43 of ticks
#     eligible, so the low action scale is not a substitute for the guard's form
#     or vice versa -- one arm needs both.
#   horizon=400. v_max=2 x horizon=200 x dt=0.04 gives only 16cm of finger
#     travel against a disengaged spawn radius of 8.0-16.1cm. The previous arm
#     ran out of horizon for a reason unrelated to its hypothesis.
#   continuous_gamma=true. Gamma_l is a target SET (sec 6.1), not a handful of
#     canonical points.
#
# NO SUCCESS BAR -- THIS IS A MEASUREMENT, and that is preregistered. But there
# is a reference: a crude scripted controller reaches ONE contact at 0.425, so
# g_one below 0.425 is a LEARNING failure, not a task failure. g_two_disp has no
# scripted reference (the controller ran out of horizon), so it measures rather
# than compares. Note the readable band for g_one is narrow -- floor 0.208 to
# reference 0.425 -- and say so when reporting.
#
# WHAT A ZERO MEANS THIS TIME. If g_two_disp also returns 0.000, the guard's
# FORM was not the wall either, and the next move is task design -- staged or
# sequential fingertip goals -- not budget. Do not re-run this at 2.4M on a flat
# zero curve. That instruction is on the record before the numbers land.
#
# REFERENCE COLUMNS REUSED, saving 6 cells: v34's gamma_free and gamma_init both
# scored 0.000 at this same 1M budget. They are not re-run.
#
# Floors regenerated 2026-09-08 by tools/make_board_v2_floor.sh BEFORE this
# sweep -- each arm moves a TASK key, so each carries its own digest AND floor:
#   g_one       digest 60c119c9ef02   floor 0.208
#   g_two_disp  digest 3b54b18fe084   floor 0.000
#
# Eq 35 accounting, MEASURED: the diagnostic eval is not free. On the g_one
# smoke it consumed 1.78x the TRAINING steps (falling as episodes start
# ending on arrival), so a 1M cell costs ~2.8M interactions, not 1M. Sweep D
# total ~17M environment interactions, plus 2 untrained floors at zero
# gradient steps. Report the 17M, not the 6M (CLAUDE.md, Reporting).
# Wall: the smoke ran 41.6 steps/s, so ~6.7h/cell at push's horizon; recontact
# runs horizon=400 so budget up to ~13h against the 18h wall.
# ===========================================================================

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/pins_gamma_ladder.sh

ARMS=(g_one g_two_disp)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# finger_velocity, not contact_frame: contact_frame is push-only (it maintains a
# contact recontact does not yet have) and ContactEnv REFUSES it rather than
# ignoring it. mask_inactive_finger=false because a two-contact Gamma needs both
# fingers to move -- with the inactive finger masked, "two contacts" is not
# reachable by construction and the arm would measure the mask, not the guard.
IFACE="obs_version=2 rich_obs=true normalize_goal_keys=false xi_gamma_mode=count \
action_interface=finger_velocity slip_model=speed_fraction slip_limit=1.0 \
mask_inactive_finger=false gap_assist=false"

case "${ARM}" in
  g_one)      GOAL="${GAMMA_GOAL_ONE}"; INIT="${GAMMA_INIT_ONE}" ;;
  g_two_disp) GOAL="${GAMMA_GOAL_TWO}"; INIT="${GAMMA_INIT_TWO}" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="recontact"
TOTAL_STEPS=1000000
# 250000, not 600000: at a 1M budget the Sweep B rung spacing would give only
# one intermediate snapshot, and the whole lesson of Sweep B is that a single
# budget can rank arms backwards.
CKPT_FREQ=250000
EVAL_EPS=32

EXTRA_OVERRIDE="${GAMMA_PINS} ${IFACE} ${GOAL} ${INIT} \
ckpt_freq=${CKPT_FREQ} diag_eval_episodes=${EVAL_EPS}"
RUN_TAG="recon_${ARM}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"
# PER-ARM PINS. Unlike Sweep C, the two arms here train on DIFFERENT tasks
# (which Gamma classes are drawn is a task key), so there is no single common
# protocol and a shared PINS.txt would be a lie. Each arm writes its own, and
# the scorer reads per arm.
printf '%s\n' "${GAMMA_PINS} ${GOAL} ${INIT}" > "${SWEEP_DIR}/.PINS.${ARM}.$$"
mv -f "${SWEEP_DIR}/.PINS.${ARM}.$$" "${SWEEP_DIR}/PINS.${ARM}.txt"

GROUP=sweepD
# No FINALIZE_TAG: finalize.sh assumes ONE common protocol for the whole sweep
# and these two arms do not share one. Score by hand, per arm, against
# PINS.<arm>.txt -- which is also the two-way treatment, since the reference
# columns are v34's and already on disk.
source slurm/_run_cell.sh
