#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=recontact_sweep
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-7%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# Superseded by the H3 speed_aware_goal work (status.md sec 7.9) -- kept for
# reference, rerun only to compare against it specifically.
#
# 8 = 2 use_her x 2 w_T x 2 seeds. w_T is a flat time penalty that opposes
# the deceleration recontact needs; zeroing it tests whether that's what's
# fighting the settle behavior, independent of the HER question.
#
# CPU-only, same as submit_test.sh/submit_sweep.sh.
# ===========================================================================

HERS=(false true); WTS=(0.0 0.02); SEEDS=(0 1)
i=$SLURM_ARRAY_TASK_ID
HER=${HERS[$((  i / 4 ))]}
WT=${WTS[$(( (i % 4) / 2 ))]}
SEED=${SEEDS[$((  i % 2 ))]}
TOTAL_STEPS=100000

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

RUN_TAG="recontact_her${HER}_wt${WT}_s${SEED}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "HER=${HER} W_T=${WT} SEED=${SEED} TOTAL_STEPS=${TOTAL_STEPS}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)"; } > "${RUN_DIR}/meta.txt"
cp "$0" "${RUN_DIR}/submit_script.sh"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_contact.py \
  contact=recontact total_steps=${TOTAL_STEPS} seed=${SEED} \
  use_her=${HER} w_T=${WT} \
  out_dir="${RUN_DIR}" \
  wandb=true wandb_run_name="${RUN_TAG}_${SLURM_ARRAY_JOB_ID}" \
  wandb_group="recontact_sweep_${SLURM_ARRAY_JOB_ID}"
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
