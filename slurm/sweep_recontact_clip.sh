#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=recontact_clip
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-10:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v21 RECONTACT sweep: where does it converge, and does the target clip stop
# the divergence?
# 12 cells = target_clip {null, 10} x seed {0..5}.   1M steps each, ~5.5h/cell
# (300k took 1h40m in job 41613939).
#
# Baseline for comparison is job 41613939 cells 4-9 (same 6 seeds, 300k, no
# clip). Everything except target_clip and total_steps is identical.
#
# BOTH ARMS ANSWER A QUESTION, so neither is filler:
#
#  target_clip=null (tasks 0-5) is the "just run longer and see where it
#  converges" experiment, and the control for the clip. Justification for
#  running it: at 300k, s0's held-out eval was still rising monotonically
#  (0.006, 0.038, 0.062, 0.119, 0.194, 0.181 per 50k) with its peak at
#  279,950 of 300,000 -- 93% of budget, i.e. budget-limited, not converged.
#  Counter-evidence: the earlier 1M run WITHOUT the done-patch (job 40910275)
#  peaked at 120-360k in all four cells then fell 4.6x by 700k, and 4 of 6
#  seeds here diverged with onset spread over 100k-250k. So the honest
#  prediction is that this arm returns ~2 usable seeds out of 6.
#
#  target_clip=10 (tasks 6-11) tests the fix. Justification: 4 of 6 seeds hit
#  critic_max 9,946-776,942 and Q reached 539 against a PROVABLE maximum of
#  10 -- with w_d=w_a=w_F=w_m=w_T=0, step_reward reduces to
#  goal_reward*reached_interface and arrival TERMINATES the episode, so there
#  is at most one +10 per episode and Q* <= goal_reward exactly. Success
#  collapsed exactly when the critic diverged (s3: critic 1 -> 810 -> 23,265
#  over 150k-300k while eval fell 0.175 -> 0.006; its final checkpoint scores
#  0/60 with the finger parked 8.06cm out). The two bounded seeds are the best
#  recontact has ever been: s0 at critic_loss 0.0, Qgap +2.8 (vs the stock
#  buffer's +44.8 at 1M) and 18/60 = 30% verified success at 0.000cm/s.
#
# WHY goal_reward AND NOT goal_reward/(1-gamma). The HER convention (OpenAI
# baselines ddpg.py: clip_by_value(r + gamma*target_Q, -clip_return, 0) with
# clip_return = 1/(1-gamma)) assumes success does NOT terminate, so the bound
# is the value of collecting the bonus every step forever. Ours terminates, so
# the tight bound is 100x smaller: 10 rather than 1000. That matters
# concretely -- the worst observed Q was 539, so a clip at 1000 would never
# have fired. Verified before submission: target_clip=null reproduces stock
# SAC bit-for-bit (identical parameter digest after 500 steps).
#
# THE REWARD IS UNCHANGED. Clipping constrains the critic's regression target,
# not the reward function. Both templates stay fully sparse (every w_* zero,
# single terminal bonus), as intended.
#
# PREREGISTERED PREDICTIONS:
#  - clip=null: ~2 of 6 seeds stay bounded and keep climbing past 300k; the
#    rest diverge and collapse. If MORE than 2 stay healthy, the divergence is
#    milder than measured and the clip matters less.
#  - clip=10: critic_max stays O(10) on all 6 seeds, and train/target_clip_frac
#    (logged by sac_clipped.py) is >0 for the seeds that would have diverged
#    and ~0 for s0/s2. If clip_frac is ~0 everywhere yet seeds still diverge,
#    the divergence is NOT target overestimation and this diagnosis is wrong.
#  - s0 under the clip should exceed its own 30% at some point past 300k,
#    since it was still rising when the last budget ran out. That is the
#    headline number to watch.
# ===========================================================================

CLIPS=(null 10); SEEDS=(0 1 2 3 4 5)
i=$SLURM_ARRAY_TASK_ID
CLIP=${CLIPS[$(( i / 6 ))]}
SEED=${SEEDS[$(( i % 6 ))]}

TEMPLATE="recontact"
TOTAL_STEPS=1000000
EXTRA_OVERRIDE="use_her=true w_T=0.0 w_a=0.0 w_m=0.0 guard_terminates=true target_clip=${CLIP}"
RUN_TAG="recontact_clip${CLIP}_s${SEED}"

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
  wandb_group="recontact_clip_${SLURM_ARRAY_JOB_ID}"
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
