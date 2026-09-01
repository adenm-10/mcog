#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=push_curric
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v32 REVERSE-CURRICULUM SWEEP -- 12 cells = 2x2 x seed{0,1,2} at 600k steps.
#
# (The previous contents were job 42300917's v29 launcher. All 24 of its per-run
# copies, logs/sweep_42300917/*/submit_script.sh, are byte-identical to the copy
# overwritten here -- verified by md5 before overwriting.)
#
# TWO QUESTIONS, one 2x2 design:
#   1. Does a reverse curriculum help?
#   2. Can it stand in for the restricted action space?
#
#            curriculum OFF   curriculum ON
#   restricted     base            curric
#   raw            raw             curric_raw
#
# Both effects get measured twice and the interaction is visible. `curric_raw`
# alone could not answer question 2 -- without `raw` you cannot tell "the
# curriculum rescued raw actions" from "raw got lucky this time".
#
# WHY THIS SHAPE OF CURRICULUM (curriculum_mode=band).
# Memo Eq 15 asks for NESTED initiation sets, I^(1) subset I^(2) subset ... A
# nested level can only DELETE far starts; it can never make near ones commoner.
# Measured (tools/probe_curriculum.py, 600 resets/level): under the nested form
# the same-room goal distance median is 2.02 / 1.94 / 2.15 / 1.78 cm across four
# levels against 2.00 cm with no curriculum at all, and cross-room only level 0
# binds. The ramp was inert, so arm `curric` would have run the same task as
# `base`.
#
# The reverse-curriculum literature does not nest. Florensa et al. (2017) grow
# start states outward from a fixed goal but KEEP ONLY those at intermediate
# difficulty, dropping mastered ones. Backplay (Resnick et al., 2018) slides a
# window backward along a demonstration. Both are MOVING WINDOWS. So `band`
# draws the GOAL first and then places the object at a distance drawn from the
# level's window -- distance becomes something we set rather than something the
# geometry hands us. This needs reset-to-arbitrary-state, which simulation gives
# for free. It is a DELIBERATE DEVIATION from Eq 15's wording, and those two
# papers are the justification; record it that way in any write-up.
#
# The windows are fractions of the distance each edge can actually reach, so
# they adapt to same-room (~0.4-22cm) and crossing (~6-16cm) with no per-edge
# constants: (0.00,0.35) (0.15,0.60) (0.35,0.85) (0.00,1.00). Width ~0.35-0.5 so
# a level is a real restriction; consecutive windows OVERLAP by ~0.2 so advancing
# is not a cliff; the LAST level is the whole range on purpose, because the
# benchmark scores every distance bin and a band final level would be a
# train/test mismatch. Measured medians by level, same-room 1.58 / 3.28 / 4.80 /
# 2.66 cm and crossing 8.55 / 10.73 / 11.76 / 9.53 cm -- a real ramp on both.
#
# THE CONTROL SHARES THE SAMPLER. curriculum_mode=band with
# curriculum_levels=null is the reverse sampler pinned at the full range and no
# schedule. So `base` and `curric` differ by the SCHEDULE ALONE, not by the reset
# distribution. Comparing a ramped band arm against the old forward sampler
# would confound the two.
#
# ADVANCEMENT. Alg 1 line 13 wants held-out LOCAL success. Two eval envs, since
# one cannot be both: the reporting env is pinned to the FULL task and never
# advances (so eval/success_rate stays comparable across arms), and a second env
# tracks the current level and is what the 0.6 threshold reads. Before
# 2026-09-01 the single eval env was built WITH curriculum_levels and nothing
# advanced it, so it sat at level 0 forever -- the gate read the EASIEST
# distribution, cleared 0.6 at once, and cleared it again at every level.
#
# PURE SPARSE, AND NO GUARD PENALTIES. Every w_* = 0, goal_reward=10 on arrival,
# arrival terminates, so Q* <= 10 exactly and target_clip=10 is a provable bound.
# guard_face=false follows from that: with no penalty term available the contact
# face could only be enforced by TERMINATION, and v31 measured that killing 72%
# of episodes at 12 ticks with all nine push cells at 0.000. Eq 40 puts the face
# in chi_push, which feeds Eq 14's -w_m(1-chi) PENALTY, and memo sec 3.1.3's
# terminating list (contact lost past n_grace, forbidden contact, off board,
# force) does not include it. So dropping it is defensible, and it is a recorded
# fidelity deviation either way.
#
# GOAL TYPES. same_room_goal_prob=0.5: half the goals are a pose in the current
# room, half a pose drawn from the doorway region. Both are scored the same way,
# by pose distance -- portal_arrival stays FALSE. Measured reason: with the
# crossing predicate ON, the untrained floor on goals >=3cm is 0.271 (0.42 in the
# two bins where crossing goals live), because a random policy shoves the object
# through a 33%-open doorway whose straight path is never blocked. With it OFF
# the floor is 0.042. The doorway POSE is the proxy for "entered the next
# region"; the real crossing test can be swapped in at eval once a policy
# exists. It also keeps ONE definition of success: the crossing predicate is not
# a point, so HER cannot relabel toward it, and rollout and relabeled rewards
# would disagree -- the bug that broke v31's recontact arms.
#
# ORIENTATION. theta_tol_deg=22.5 is Eq 11's eight orientation bins (360/8 = 45
# deg bins = +/-22.5). v31 measured orientation was NOT push's binding term --
# 34/60 episodes reached the window while only 1/60 reached position -- so it is
# kept because it is cheap and faithful, not because it is expected to bite.
#
# WHAT THE v29/v31 EVIDENCE PUT BACK. mask_inactive_finger=true (v29: unmasking
# is a wash on success and 4x worse on forbidden_contact), push_cone_deg=30 (v30:
# 30->90 costs 0.822->0.767), push_range_min_cm=null (v28: the 3cm floor bought
# exactly 0.00, and it fights the window's near end), object_theta_spread_deg=
# null, angular_drag_arm_cm=3.12 (the derived uniform-pressure value; 6.00 is
# above the physical ceiling), gap_assist=false (v29 nogapassist, +0.083).
#
# PHASE 0 ALREADY RAN, at zero training cost:
#   tools/probe_curriculum.py --mode band   PASS, 0 leaks, every level restricts
#   tools/make_v32_floor.sh                 logs/eval/v32_floor/, 4 cells
# UNTRAINED FLOOR on this exact protocol, goals >=3cm:
#   restricted actions 0.042    raw actions 0.000
#
# PREREGISTERED VERDICTS -- written before the sweep, not after.
#   The curriculum HELPS if curric > base AND curric_raw > raw on goals >=3cm,
#   on >=2 of 3 seeds, under BOTH model and model_best. (model_best inverted the
#   whole arm ordering on v29's gap-assist result, so one checkpoint is not
#   enough.)
#   The curriculum REPLACES the action restriction if curric_raw clears its
#   0.000 floor on >=2 seeds AND reaches base. Anything less means the
#   restriction is still doing the work.
#
# SCORING. tools/score_sweep.py with --pins set to the PROTO line below, which
# is byte-identical to logs/eval/v32_floor/PROTOCOL.md. Report goals >=3cm
# BESIDE the 5-bin mean: the 0-3cm bin has a nonzero floor and halves every arm
# difference. Check the printed env digest matches the floor's before comparing.
#
# NOT IN THIS SWEEP, deliberately:
#   require_settled. It is real spec (Eq 13 bounds object speed) and it is what
#   makes push composable, but v28 measured 53% of failures already land within
#   1cm, so turning it on now would add a second hard term to a task whose first
#   one is unproven. Its own arm, later.
#   The adaptive window (Florensa's actual rule: keep the window where success
#   sits between ~10% and ~90%, rather than advancing on a fixed threshold).
#   More machinery, and only worth it if the fixed schedule shows something.
#   The object-frame observation fix. Fingertip positions are object-relative but
#   world-oriented while wall distances are already object-frame. It has to move
#   the ACTION frame too and it strands every checkpoint. Its own change.
# ===========================================================================

set -e

ARMS=(base curric raw curric_raw)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# portals=[{...}] must reach Hydra through a shell variable: bash brace-expands
# [{a,b,c}] into three words, in heredocs and sbatch scripts alike.
PORT="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"

# Stated explicitly in every arm, never left to a default: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
# PROTO must stay byte-identical to logs/eval/v32_floor/PROTOCOL.md.
PROTO="require_settled=false push_cone_deg=30 same_room_goal_prob=0.5 \
push_range_min_cm=null object_theta_spread_deg=null angular_drag_arm_cm=3.12 \
portal_arrival=false portal_goal=true portal_clearance_cm=0.5 \
guard_face=false rich_obs=true push_range_max_cm=null"
GOAL="theta_tol_deg=22.5 theta_goal_window_deg=45.0"
IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 mask_inactive_finger=true gap_assist=false"
# band + no levels = the reverse sampler at the full range, no schedule.
CURRIC="curriculum_mode=band curriculum_levels=null"

case "${ARM}" in
  base)       ;;
  curric)     CURRIC="curriculum_mode=band curriculum_levels=4 curriculum_threshold=0.6" ;;
  raw)        IFACE="action_interface=finger_velocity mask_inactive_finger=true" ;;
  curric_raw) IFACE="action_interface=finger_velocity mask_inactive_finger=true"
              CURRIC="curriculum_mode=band curriculum_levels=4 curriculum_threshold=0.6" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=600000
CKPT_FREQ=200000
# 32, not the default 16: the advance gate compares a success rate against 0.6,
# and 16 episodes make that a 10-of-16 coin flip.
EVAL_EPS=32
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
${PORT} min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 \
target_clip=10 ckpt_freq=${CKPT_FREQ} diag_eval_episodes=${EVAL_EPS} \
board_w_cm=50.0 board_h_cm=30.0 disengaged_away_deg=60 \
${PROTO} ${GOAL} ${CURRIC} ${IFACE}"
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
  # whether the tree was dirty and a hash of the diff, so a run's provenance
  # is checkable rather than a guess (docs/PROGRESS.md, v20).
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
  wandb_group="curric_${SLURM_ARRAY_JOB_ID}"
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
