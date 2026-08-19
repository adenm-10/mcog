#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=warmup_sameroom_zerovel
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# Tests this session's three new mechanisms (status.md sec 7.9's follow-ups),
# each gate-verified (bit-identical at defaults, or a direct sampled-batch
# check) before this sweep was written.
#
# Tasks 0-1: push, learning_starts 250 -> 10000 (this sweep's earlier
# 100k-budget runs silently defaulted to 250 -- barely one episode -- unlike
# the deliberate 2000 the earlier w_m guard sweep used; 10k gives ~50
# diverse episodes in the buffer before any gradient step, instead of ~1).
# Both cells also get same_room_goal_prob wired in (domains/contact/
# gym_env.py, config/train_contact.yaml, train_contact.py) -- push's real
# (reset-time) goal was always cross-room, but HER can only ever relabel
# from positions the object actually reached, which for this task stays
# inside the source room 94/100 episodes measured -- i.e. HER already
# implicitly trains same-room reachability almost exclusively, while the
# real desired_goal never matches that. same_room_goal_prob draws the goal
# from the source room itself some fraction of the time, closing that gap
# rather than asking the network to generalize almost entirely
# out-of-distribution to the cross-room case.
#   Task 0: same_room_goal_prob=0.0 -- control, isolates the warmup bump
#     alone (same-room off, otherwise identical to task 1).
#   Task 1: same_room_goal_prob=0.5.
#   Both: min_progress_cm=3.0, her_n_sampled_goal=4 (this session's sweep,
#     job 40325825, found cutting both min_progress_cm and n_sampled_goal
#     together hurts real behavior at 100k -- keeping n_sampled_goal at its
#     default here rather than compounding cuts). 150k budget, not 100k --
#     10k of pure-random warmup is 10% of a 100k run and wasn't in the
#     original budget's accounting; 150k keeps warmup down to ~6.7% and
#     gives real training time comparable to the earlier 100k runs plus
#     margin for the now-more-diverse goal distribution to shake out.
#
# Task 2: recontact, ZeroVelocityGoalHerReplayBuffer (domains/contact/
# her_buffer.py, new this session) -- HER's relabeled desired_goal carries
# whatever real speed happened to occur at the picked future tick into the
# actor/critic's own INPUT features (not just the reward calc, which
# already ignored it), while a real reset always fixes desired_goal's speed
# to exactly 0.0 -- a train/rollout input mismatch, confirmed by reading
# HerReplayBuffer._get_virtual_samples directly. The fix pins the relabeled
# goal's velocity slot to 0.0, verified on a live sampled batch
# (desired_goal[:,2] all exactly 0.0, achieved_goal[:,2] still real/varying)
# before this sweep was written. Wired in automatically whenever
# speed_aware_goal=true -- no separate flag needed.
#   1,000,000 steps -- "generous, so the test is fair" per explicit
#   instruction: double the 500k budget already run (which showed zero
#   movement on the blocking metric, finger speed at closest approach,
#   across 200k/500k and every shaping variant tried), so a null result
#   here can't be attributed to under-training.
# ===========================================================================

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 2 ]; then
  TEMPLATE="push"
  SAME_ROOM_PROBS=(0.0 0.5)
  SRP=${SAME_ROOM_PROBS[$i]}
  TOTAL_STEPS=150000
  EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=false board_w_cm=50.0 board_h_cm=30.0 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}] min_progress_cm=3.0 her_n_sampled_goal=4 learning_starts=10000 same_room_goal_prob=${SRP}"
  RUN_TAG="push_warmup10k_sameroom${SRP}"
else
  TEMPLATE="recontact"
  TOTAL_STEPS=1000000
  EXTRA_OVERRIDE="use_her=true speed_aware_goal=true w_T=0.0 w_a=0.0 w_m=0.0"
  RUN_TAG="recontact_zerovel_sparse_1M"
fi

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "TEMPLATE=${TEMPLATE} TOTAL_STEPS=${TOTAL_STEPS} EXTRA_OVERRIDE=${EXTRA_OVERRIDE}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)"; } > "${RUN_DIR}/meta.txt"
cp "$0" "${RUN_DIR}/submit_script.sh"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_contact.py \
  contact=${TEMPLATE} total_steps=${TOTAL_STEPS} seed=0 \
  ${EXTRA_OVERRIDE} \
  out_dir="${RUN_DIR}" \
  wandb=true wandb_run_name="${RUN_TAG}_${SLURM_ARRAY_JOB_ID}" \
  wandb_group="warmup_sameroom_zerovel_${SLURM_ARRAY_JOB_ID}"
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
