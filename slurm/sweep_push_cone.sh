#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=push_cone
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v21 PUSH sweep: does training on a WELL-POSED push task work?
# 12 cells = push_cone_deg {null, 30, 45} x same_room_goal_prob {0.5, 1.0}
#            x seed {0, 1}.   150k steps each, ~1h/cell.
#
# Baseline for comparison is job 41613939 cells 0-3. Everything except
# push_cone_deg and same_room_goal_prob is byte-identical to that run's
# override, so the sampler is the only thing that moved.
#
# WHY. v21 measured that the historical sampler draws the contact face
# independently of where the goal is, so the median episode needed the object
# pushed ~98 deg away from the only direction the finger could push it, 56% of
# episodes needed a push that moves the object AWAY from the goal, and only
# ~5% were solvable by any straight push. No hand-coded controller beat doing
# nothing on that distribution (all at 1.0%, Fisher p=0.62 vs inaction), and
# neither did the trained policies. Solving the rest needs walking the finger
# to another face, which push_guard forbids (5-tick grace allows 4.0cm of
# travel; rounding a 10x6cm object needs ~7.5cm) -- i.e. it is a
# push->recontact->push task being trained as one edge.
#
# push_cone_deg draws face and goal CONSISTENTLY (memo Eq 7 makes the object
# face an edge parameter; Eq 12 puts the finger on the west face for an
# eastward push). Cross-room the face is determined and the goal is coned
# inside dst through the portal; same-room the face is free so all four are
# sampled and the goal is coned to match, which keeps the face diversity
# Eq 9's shared network needs.
#
# THE RESULT THIS ARM IS FOLLOWING UP. The ALREADY-TRAINED policies from job
# 41613939 score 34-39% on the coned sampler with ZERO retraining, against
# 1-2% on the distribution they were trained on (200 deterministic episodes;
# do-nothing scores 0.000). So the skill was already there and ~95% of
# training episodes were unsolvable by anything. This tests whether training
# on the solvable distribution compounds that.
#
# PREREGISTERED PREDICTIONS:
#  - cone=null cells reproduce ~1-2% success (they are the control).
#  - cone=30/45 cells clear 34-39% -- the zero-retraining transfer is the
#    floor, not the target, since these actually train on it.
#  - srg=1.0 (median goal 2cm) beats srg=0.5 (median 16cm), because once
#    direction is fixed the long-range task is PRECISION-limited: at 21cm even
#    12 deg of misalignment misses by 4.7cm against arrival_eps=0.4. If
#    srg=0.5 does NOT lag, range matters less than predicted and Eq 15's
#    range ramp drops in priority.
#  - contact retention stays the deeper limit either way: the trained
#    policies hold contact 43-47% of ticks against a scripted rule's 100%.
#
# NOT changed here, deliberately: target_clip stays null. Push's critic did
# not diverge last run (max 12.8-20.5), and adding the clip would confound
# the sampler test. Reward stays fully sparse (every w_* zero).
# ===========================================================================

CONES=(null 30 45); SRGS=(0.5 1.0); SEEDS=(0 1)
i=$SLURM_ARRAY_TASK_ID
CONE=${CONES[$((  i / 4 ))]}
SRG=${SRGS[$(( (i % 4) / 2 ))]}
SEED=${SEEDS[$((  i % 2 ))]}

TEMPLATE="push"
TOTAL_STEPS=150000
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true board_w_cm=50.0 board_h_cm=30.0 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}] min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 same_room_goal_prob=${SRG} push_cone_deg=${CONE} require_settled=false target_clip=null"
RUN_TAG="push_cone${CONE}_srg${SRG}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "TEMPLATE=${TEMPLATE} TOTAL_STEPS=${TOTAL_STEPS} EXTRA_OVERRIDE=${EXTRA_OVERRIDE}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)";
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
  wandb_group="push_cone_${SLURM_ARRAY_JOB_ID}"
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
